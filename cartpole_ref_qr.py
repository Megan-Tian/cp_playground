from datetime import datetime
import warnings

import utils
warnings.filterwarnings("ignore")
from torch import multiprocessing


from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import tensordict
import torchrl
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger
from torchrl.envs import TransformedEnv, DMControlEnv

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 5000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

env_name = "InvertedPendulum-v5"
# env_name = "InvertedDoublePendulum-v4"

def train():
    base_env = GymEnv(env_name, device=device)

    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )

    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    # check env specs
    # print("normalization constant shape:", env.transform[0].loc.shape)
    # print("observation_spec:", env.observation_spec)
    # print("reward_spec:", env.reward_spec)
    # print("input_spec:", env.input_spec)
    # print("action_spec (as defined by input_spec):", env.action_spec)
    check_env_specs(env)

    # rollout = env.rollout(3)
    # print("rollout of three steps:", rollout)
    # print("Shape of the rollout TensorDict:", rollout.batch_size)

    ################ POLICY ################
    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )

    ################ VALUE NET #################
    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    print("Running policy:", policy_module(env.reset()))
    print("Running value:", value_module(env.reset()))

    ################ DATA COLLECTOR & REPLAY BUFFER ################
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    ################ PPO LOSS ################
    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True, device=device,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )


    ###################### TRAIN LOOP ######################
    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""
    writer = SummaryWriter(f"runs/{env_name}_ref_pytorch_ppo_{total_frames}")

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for epoch in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # tensorboard logging
                writer.add_scalar("loss/train_loss_objective", loss_vals["loss_objective"], epoch + i*num_epochs)
                writer.add_scalar("loss/train_loss_critic", loss_vals["loss_critic"], epoch + i*num_epochs)
                writer.add_scalar("loss/train_loss_entropy", loss_vals["loss_entropy"], epoch + i*num_epochs)
                writer.add_scalar("loss/train_total", loss_value, epoch + i*num_epochs)
                # writer.add_scalar("Loss/train_q_low_loss", q_low_loss.mean().item(), epoch + i*num_epochs)
                # writer.add_scalar("Loss/train_q_high_loss", q_high_loss.mean().item(), epoch + i*num_epochs)

                # coverage = torch.mean(((tensordict_data["action"] >= tensordict_data['q_low'].squeeze()) & (tensordict_data["action"] <= tensordict_data['q_high'].squeeze())).type(torch.float32)).float()
                # writer.add_scalar("Loss/action_coverage_pct", coverage, epoch + i*num_epochs)

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

        # tensorboard logging
        writer.add_scalar("train/training_reward_avg", logs["reward"][-1], i*num_epochs)
        writer.add_scalar("train/training_max_step_count", logs["step_count"][-1], i*num_epochs)
        writer.add_scalar("train/training_lr", logs["lr"][-1], i*num_epochs)

        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )

                # tensorboard logging
                writer.add_scalar("eval/eval_reward_cumulative", logs["eval reward (sum)"][-1], i*num_epochs)
                writer.add_scalar("eval/eval_reward_avg", logs["eval reward"][-1], i*num_epochs)
                writer.add_scalar("eval/eval_max_step_count", logs["eval step_count"][-1], i*num_epochs)

                del eval_rollout

        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()


    # save policy, training logs
    torch.save(policy_module, f'baseline/{env_name}-policy-{total_frames}.pth')
    print(f'saved policy!')
    torch.save(logs, f'baseline/{env_name}-train-val-logs-{total_frames}.pkl')


    writer.flush()
    writer.close()





def eval():
    ################## LOAD POLICY ##################
    policy_path = f"baseline/{env_name}-policy-{total_frames}.pth"
    print(f'Loading policy from {policy_path}')
    policy_module = torch.load(policy_path, map_location=torch.device("cpu"), weights_only=False)
    policy_module.eval()
    
    ################## SETUP ENV ##################
    env = TransformedEnv(
        GymEnv(env_name, device="cpu", from_pixels=True, pixels_only=False),
        Compose(
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0) # initialize the env transforms

    # print("normalization constant shape:", env.transform[0].loc.shape)
    # print("observation_spec:", env.observation_spec)
    # print("reward_spec:", env.reward_spec)
    # print("input_spec:", env.input_spec)
    # print("action_spec (as defined by input_spec):", env.action_spec)
    print("Running policy:", policy_module(env.reset()))
    check_env_specs(env)

    ################## ROLLOUT (using torchrl utils) ##################
    # save a video of the rollout
    # logger = CSVLogger(exp_name="{env_name}", log_dir="{env_name}_videos", video_format="mp4")
    # env = TransformedEnv(env, VideoRecorder(logger=logger, tag=f"run_video_{datetime.now()}", in_keys=["pixels"], fps=30))

    # Evaluate policy
    # with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    #     rollout = env.rollout(max_steps=1000, policy=policy_module)

    # save rollout video
    # env.transform.dump()
    # print("Video should now be saved in:", logger.log_dir)
    
    ################## ROLLOUT (manual stepping) ##################
    # preallocate:
    max_ep_steps = 1000
    rollout = tensordict.TensorDict(batch_size=[max_ep_steps])
    # reset
    _data = env.reset()
    i = 0
    done = False
    while i < max_ep_steps and not done:
        action = policy_module(_data.to_tensordict())
        print(f'action shape = {action.shape} | action = {action} | _data["terminated"] = {_data["terminated"]}')
        _data["action"] = action.get("action")
        _data = env.step(_data) # advance env
        # rollout.set_at_(_data, index=i)
        rollout[i] = _data
        _data = torchrl.envs.utils.step_mdp(_data, keep_other=True) # advance data tensordict
        done = _data["terminated"].item() or _data["truncated"].item() or _data["done"].item()
        i += 1
    # return data
    print(f'quit before step {i}')
    rollout = rollout[:i]

    utils.video_writer(rollout["pixels"], f"{env_name}_videos/{env_name}-{datetime.now()}.mp4")

    ################## PLOT RESULTS ##################
    rewards = rollout["next", "reward"]
    actions = rollout["action"]
    observations = rollout["observation"]
    print(f"rewards.shape: {rewards.shape} | actions.shape: {actions.shape} | observations.shape: {observations.shape}")
    total_reward = rewards.sum().item()
    step_count = rollout["step_count"].max().item()

    print(f"Eval complete")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Steps survived: {step_count}")
    print(f"rollout keys: {rollout.keys()}")
    print(f"rollout['pixels'].shape: {rollout['pixels'].shape}")

    # reward per step
    plt.plot(rewards.squeeze().numpy(), label="reward")
    plt.title("Reward per Step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # actions + obs
    plt.plot(actions.detach().numpy(), label="action", marker="o")
    for i in range(observations.shape[1]):
        plt.plot(observations[:, i].numpy(), label=f"observation[{i}]")

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    env.close()


if __name__ == "__main__":
    # train()
    eval()



