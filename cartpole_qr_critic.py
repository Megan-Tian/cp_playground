from datetime import datetime
import warnings

import numpy as np

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
import torch.nn.functional as F  
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
total_frames = 100_000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = 0.2  # clip value for PPO loss

gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

### MODIFIED ###
# Define quantile levels for the auxiliary loss
low_quantile = 0.1  # The 10th percentile
high_quantile = 0.9 # The 90th percentile
aux_loss_coef = 1 # Weight for the auxiliary quantile loss

env_name = "InvertedPendulum-v5"
# env_name = "InvertedDoublePendulum-v4"

def pinball_loss(prediction, target, quantile):
    """
    Calculates the pinball loss between `prediction` and `target` for a given `quantile` between (0, 1).
    """
    loss = torch.where(
        target >= prediction,
        quantile * (target - prediction),
        (1 - quantile) * (prediction - target),
    )
    return loss.mean()


### MODIFIED ###
# Create a custom module to handle the new actor head
class QuantileActorHead(nn.Module):
    """
    A custom module that wraps the actor network.
    
    The network outputs a tensor of shape [B, 4 * action_dim].
    This module splits that tensor into four parts:
    - loc (mean)
    - log_scale (log std)
    - q_low (low quantile prediction)
    - q_high_delta_raw (a raw value for the high quantile's positive offset)
    
    It then processes log_scale into scale (std) by exponentiating it
    and processes q_high_delta_raw into a positive value. 
    to compute q_high, guaranteeing q_high > q_low.
    """
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Network outputs a single tensor with 4*action_dim features
        raw_output = self.network(observation)
        
        # TOGGLE THIS TO PRED Q_LOW + DELTA_HIGH vs DELTA_LOW + DELTA_HIGH
        use_offset_from_action = True
        
        if use_offset_from_action:
            loc, log_scale, q_low_delta_raw, q_high_delta_raw = raw_output.chunk(4, dim=-1)
            # scale = torch.exp(log_scale)
            scale = tensordict.nn.mappings("biased_softplus_1.0")(log_scale)
            q_low = loc - (F.softplus(q_low_delta_raw))
            q_high = loc + (F.softplus(q_high_delta_raw))
            # q_low = loc + q_low_delta_raw
            # q_high = loc + q_high_delta_raw
            
        else:
            # split output into four equal parts along the last dimension
            # We now interpret the last chunk as the *raw offset* for q_high
            # loc, log_scale, q_low, q_high_delta_raw = raw_output.chunk(4, dim=-1)
            
            # Process scale: std must be positive. ProbabilisticActor returns log probs
            # scale = torch.exp(log_scale)
            # scale = tensordict.nn.mappings("biased_softplus_1.0")(log_scale)
            
            # process the high quantile to be strictly greater than the low quantile
            # q_high_delta = F.softplus(q_high_delta_raw) + 1e-6
            # q_high = q_low + q_high_delta
            
            
            # pred low and high directly
            loc, log_scale, q_low, q_high = raw_output.chunk(4, dim=-1)
            scale = tensordict.nn.mappings("biased_softplus_1.0")(log_scale)
            
            
        return loc, scale, q_low, q_high


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
    check_env_specs(env)

    ################ POLICY ################
    
    ### MODIFIED ###
    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        # final layer now outputs 4 * action_dim
        nn.LazyLinear(4 * env.action_spec.shape[-1], device=device),
        # REMOVED NormalParamExtractor()
    )

    # wrap the base network in our custom head module
    custom_head_module = QuantileActorHead(actor_net)

    # wrap the custom module in TensorDictModule
    policy_module = TensorDictModule(
        module=custom_head_module,
        in_keys=["observation"],
        # added new output keys q_low, q_high
        out_keys=["loc", "scale", "q_low", "q_high"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,

        # ProbabilisticActor *only* uses loc and scale, set that here
        in_keys=["loc", "scale"],
        
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
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
    writer = SummaryWriter(f"runs/{env_name}_quantile_ppo_{total_frames}")

    for i, tensordict_data in enumerate(collector):
        for epoch in range(num_epochs):
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())

            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size).to(device)
                
                # PPO LOSS
                loss_vals = loss_module(subdata)
                
                # AUXILIARY QUANTILE LOSS
                action_target = subdata["action"].detach() 
                
                q_low_pred = subdata["q_low"]
                q_high_pred = subdata["q_high"]

                loss_q_low = pinball_loss(q_low_pred, action_target, low_quantile)
                loss_q_high = pinball_loss(q_high_pred, action_target, high_quantile)
                
                aux_loss = (loss_q_low + loss_q_high) * aux_loss_coef
                
                # TOTAL LOSS
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                    + aux_loss  # Add the auxiliary loss
                )

                # tensorboard logging
                writer.add_scalar("loss/train_loss_objective", loss_vals["loss_objective"].item(), epoch + i*num_epochs)
                writer.add_scalar("loss/train_loss_critic", loss_vals["loss_critic"].item(), epoch + i*num_epochs)
                writer.add_scalar("loss/train_loss_entropy", loss_vals["loss_entropy"].item(), epoch + i*num_epochs)
                writer.add_scalar("loss/train_total", loss_value.item(), epoch + i*num_epochs)
                writer.add_scalar("loss/train_q_low_loss", loss_q_low.item(), epoch + i*num_epochs)
                writer.add_scalar("loss/train_q_high_loss", loss_q_high.item(), epoch + i*num_epochs)
                writer.add_scalar("loss/train_q_loss_total", aux_loss.item(), epoch + i*num_epochs)

                # Calculate coverage metric (action within predicted bounds)
                with torch.no_grad():
                    action_in_range = (subdata["action"] >= subdata["q_low"]) & (subdata["action"] <= subdata["q_high"])
                    coverage = action_in_range.float().mean()
                    writer.add_scalar("metrics/action_coverage_pct", coverage.item(), epoch + i*num_epochs)

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()
        
        # terminal logging / prints during training
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
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
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

                writer.add_scalar("eval/eval_reward_cumulative", logs['eval reward (sum)'][-1], i*num_epochs)
                writer.add_scalar("eval/eval_reward_avg", logs["eval reward"][-1], i*num_epochs)
                writer.add_scalar("eval/eval_max_step_count", logs["eval step_count"][-1], i*num_epochs)

                del eval_rollout

        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
        scheduler.step()


    torch.save(policy_module, f'baseline/{env_name}-policy-{total_frames}.pth')
    print(f'saved policy!')
    torch.save(logs, f'baseline/{env_name}-train-val-logs-{total_frames}.pkl')

    writer.flush()
    writer.close()


def eval(num_episodes = 10, conformalized=False, q_hat=None):
    ################## LOAD POLICY ##################
    policy_path = f"baseline/{env_name}-policy-{total_frames}.pth"
    # policy_path = f"inverted_pendulum_100k_ckpt/InvertedPendulum-v5-policy-100000.pth"
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
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    
    ### MODIFIED ###
    print("Initializing lazy layers...")
    policy_module(env.reset())
    print("Lazy layers initialized.")
    
    print("Running policy:", policy_module(env.reset()))
    check_env_specs(env)
    
    all_actions = []
    all_scales = []
    all_q_lows = []
    all_q_highs = []
    

    for _ in range(num_episodes):
        ################## ROLLOUT (manual stepping) ##################
        max_ep_steps = 50
        rollout = tensordict.TensorDict(batch_size=[max_ep_steps])
        
        _data = env.reset()
        i = 0
        done = False

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            while i < max_ep_steps and not done:
                # policy_module returns a tensordict with 'action', 'loc', 'scale', 'q_low', 'q_high'
                action_td = policy_module(_data.to_tensordict())
                
                # only need to assign the 'action' key for the env step
                _data["action"] = action_td.get("action")
                
                _data = env.step(_data) # advance env
                
                # Store ALL data from the step (including 'q_low' and 'q_high' from action_td)
                # We merge the action_td into the _data tensordict before saving
                rollout[i] = _data.clone().update(action_td)
                
                _data = torchrl.envs.utils.step_mdp(_data, keep_other=True) # advance data tensordict

                done = _data["terminated"].item() or _data["truncated"].item() or _data["done"].item()
                i += 1
                
        print(f'Rollout finished at step {i}')
        rollout = rollout[:i] # Trim the rollout to the actual length

        utils.video_writer(rollout["pixels"], f"{env_name}_videos/{env_name}-{datetime.now()}.mp4")

        ################## PLOT RESULTS ##################
        rewards = rollout["next", "reward"]
        actions = rollout["action"]
        observations = rollout["observation"]
        
        ### MODIFIED ###
        # Extract the new quantile keys
        q_low = rollout["q_low"]
        q_high = rollout["q_high"]
        scale = rollout["scale"]
        
        print(f"rewards.shape: {rewards.shape} | actions.shape: {actions.shape} | observations.shape: {observations.shape}")
        print(f"q_low.shape: {q_low.shape} | q_high.shape: {q_high.shape}")
        
        total_reward = rewards.sum().item()
        step_count = rollout["step_count"].max().item()

        print(f"Eval complete")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Steps survived: {step_count}")
        print(f"rollout keys: {rollout.keys()}")
        # print(f"rollout['pixels'].shape: {rollout['pixels'].shape}")

        # reward
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1) # Top plot
        plt.plot(rewards.squeeze().cpu().numpy(), label="reward")
        plt.title(f"Reward per Step (Total: {total_reward:.2f})")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid()
        plt.legend()

        # actions / obs / quantiles
        plt.subplot(3, 1, 2) # middle plot
        
        # actions
        action_np = actions.detach().cpu().numpy().squeeze()
        steps = range(len(action_np))
        plt.plot(steps, action_np, label="action (deterministic)", marker="o", markersize=2, linestyle='None')

        # quantile predictions
        q_low_np = q_low.detach().cpu().numpy().squeeze()
        q_high_np = q_high.detach().cpu().numpy().squeeze()
        
        if conformalized and q_hat is not None:
            # conform_pred is a function (q_low, q_high, q_hat) --> (conformal q_low, conformal q_high)
            q_low_np = q_low_np - q_hat
            q_high_np = q_high_np + q_hat
        
        coverage = ((actions >= q_low) & (actions <= q_high)).float().mean().item()
        print(f"Action quantile coverage: {coverage * 100:.2f}%")
        plt.plot(steps, q_low_np, label=f"q_low ({low_quantile*100:.0f}th percentile)", color='orange', linestyle='--')
        plt.plot(steps, q_high_np, label=f"q_high ({high_quantile*100:.0f}th percentile)", color='green', linestyle='--')
        
        # fill in area between quantiles
        plt.fill_between(
            steps,
            q_low_np,
            q_high_np,
            color='gray', alpha=0.2, label="Predicted Quantile Band"
        )
        
        scale_np = scale
        print(f'scale shape = {scale_np.shape}')
        plt.plot(steps, actions - scale_np, label="std of action", color='grey')
        plt.plot(steps, actions + scale_np, label="std of action", color='grey')
        # plt.fill_between(
        #     steps,
        #     actions - scale,
        #     actions + scale,
        #     color='blue', alpha=0.2, label="Action Band"
        # )
        
        # Plot observations
        obs_np = observations.cpu().numpy()
        # for j in range(obs_np.shape[1]):
        #     plt.plot(steps, obs_np[:, j], label=f"observation[{j}]", alpha=0.5)

        plt.title("Actions and Observations")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Legend outside

        
        # plot the difference between quantiles
        plt.subplot(3, 1, 3) # bottom plot
        plt.plot(steps, q_high_np - q_low_np, label="Q_high - Q_low")
        plt.plot(steps, action_np - q_low_np, label="Action - Q_low")
        plt.plot(steps, q_high_np - action_np, label="Q_high - Action")
        plt.plot(steps, scale_np, label="scale")
        plt.title("Quantile Differences")
        plt.xlabel("Step")
        plt.ylabel("Value")
        
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Legend outside
        plt.tight_layout()
        plt.show()
        
        all_actions.append(action_np)
        all_q_lows.append(q_low_np)
        all_q_highs.append(q_high_np)
        all_scales.append(scale)

    env.close()

    return all_actions, all_q_lows, all_q_highs, all_scales
    # CQ
    # https://mlwithouttears.com/2024/01/17/conformalized-quantile-regression/ bullet 3)
    # scores = max(actions - q_high, q_low - actions)
    # alpha = 0.1
    # q_cqr = np.quantile(scores, 1-alpha)
    # coverage_interval_low = q_low - q_cqr
    # coverage_interval_high = q_high + q_cqr
    # coverage = ((actions >= coverage_interval_low) & (actions <= coverage_interval_high)).float().mean().item()
    # print(f"Action quantile coverage: {coverage * 100:.2f}%")


if __name__ == "__main__":
    # train()
    eval()
    # eval()
    # eval()

