# thanks https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html

import json
import warnings

import numpy as np
warnings.filterwarnings("ignore")
from torch import multiprocessing

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
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

from utils import ActorWithQuantiles, quantile_loss

# setup
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
np.random.seed(0)
torch.manual_seed(0)

num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

# data collection params
frames_per_batch = 1000
# for complete training, bring the number of frames up to 1M
total_frames = 500_000 # 100k takes 3-4min to on laptop 4070 gpu

# PPO params
sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

# QR params
alpha = 0.1 # setting convention here that alpha = lower quantile

# setup env
base_env = GymEnv("InvertedDoublePendulum-v4", device=device)

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

print("normalization constant shape:", env.transform[0].loc.shape)
print("observation_spec:", env.observation_spec)
print("reward_spec:", env.reward_spec)
print("input_spec:", env.input_spec)
print("action_spec (as defined by input_spec):", env.action_spec)

check_env_specs(env)

# policy
actor_net_backbone = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),

    # +2 is for lower and upper quantiles
    # 2 * action dim is for "loc" (mean) and "scale" (var) outputs
    nn.LazyLinear(2 * env.action_spec.shape[-1] + 2, device=device), 
)

actor_net = ActorWithQuantiles(
    actor_net_backbone,
    action_dim=env.action_spec.shape[-1],
    action_low=env.action_spec.space.low,
    action_high=env.action_spec.space.high,
    device=device
)


print(f'actor net spec: {actor_net}')


policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale", "q_low", "q_high"]
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
)

# value net
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

# data collector, data buffer
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

# PPO loss
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


print(f"action_dim: {env.action_spec.shape[-1]}")
print(f"Final layer output features: {4 * env.action_spec.shape[-1]}")

print(f"actor_net.action_dim: {actor_net.action_dim}")

# Test forward
test_input = torch.randn(5, 11).to(device)
intermediate = actor_net_backbone(test_input)
print(f"Backbone output shape: {intermediate.shape}")
print(f"Policy out would be: {intermediate[..., :2*actor_net.action_dim].shape}")
print(f"Quantile out would be: {intermediate[..., 2*actor_net.action_dim:].shape}")

loc, scale, q_low, q_high = actor_net(test_input)
print(f"q_high shape: {q_high.shape}")
print(f"q_low shape: {q_low.shape}")
print('------------------------------------------------------------------------------')


###################### TRAIN LOOP ######################
logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

for i, tensordict_data in enumerate(tqdm(collector)):
    # now work with a specific batch of data
    for _ in range(num_epochs):
        # PPO advantage signal, depends on value network which is updated in the inner loop
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())

        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            # loss module also updates value net
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # quantile loss
            q_low_loss, q_high_loss = quantile_loss(
                q_low=alpha, 
                q_high=(1-alpha), 
                actions=tensordict_data["action"], 
                alpha=alpha
            )

            # print(f'q low loss: {q_low_loss.shape}')
            # print(f'q high loss: {q_high_loss.shape}')
            # print(f'loss value: {loss_value}')
            # print(f"q_low range: [{subdata['q_low'].min():.3f}, {subdata['q_low'].max():.3f}]")
            # print(f"q_high range: [{subdata['q_high'].min():.3f}, {subdata['q_high'].max():.3f}]")
            # print(f"action range: [{subdata['action'].min():.3f}, {subdata['action'].max():.3f}]")
            # print(f"q_loss: {q_low_loss.mean().item()}, {q_high_loss.mean().item()}")
            # print(f'loss val before combining with q loss: {loss_value}')

            # combine losses
            loss_lambda = 1 # TODO properly tune this hyperparam
            loss_value = loss_value + loss_lambda * (q_low_loss.mean() + q_high_loss.mean())

            # Backward pass: grad clipping and optimization step
            loss_value.backward()
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

    if i % 10 == 0:
        print(f'Evaluating policy after {i} frames!')
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            # print(f'eval_rollout: {eval_rollout}')
            
            logs["eval_reward_step_avg"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval_reward_episode_total"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval_step_count"].append(eval_rollout["step_count"].max().item())

            # log quantile coverage
            actions = eval_rollout["action"]
            q_low = eval_rollout["q_low"]
            q_high = eval_rollout["q_high"]
            print(f'actions shape: {actions.shape}, q_low shape: {q_low.shape}, q_high shape: {q_high.shape}')

            # TODO it's giving !SFB !RFC byeeee
            if actions is not None:
                coverage = torch.mean(((actions >= q_low) & (actions <= q_high)).type(torch.float32)).float()
                logs["eval_coverage"].append(coverage) # raw fraction of covered actions for current rollout
            else:
                print(f'WARNING: actions is None')
                print(f'actions: {actions}')
                print(f'q_low: {q_low}')
                print(f'q_high: {q_high}')
            
            logs["eval_actions"].append(actions)
            logs["eval_q_low"].append(q_low) # log q_low
            logs["eval_q_high"].append(q_high)

            eval_str = (
                f"eval cumulative reward: {logs['eval_reward_episode_total'][-1]: 4.4f} "
                f"(init: {logs['eval_reward_episode_total'][0]: 4.4f}), "
                f"eval step-count: {logs['eval_step_count'][-1]}"
            )
            del eval_rollout

    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
    scheduler.step()

# save policy
torch.save(policy_module, 'cartpole-policy.pth')
print(f'saved policy!')

# results yay
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
# training
plt.plot(logs["reward"])
plt.title("Training rewards (avg per step)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
# validation
plt.subplot(2, 2, 3)
plt.plot(logs["eval_reward_step_avg"])
plt.title("Validation rewards (avg per step)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval_step_count"])
plt.title("Max step count (validation)")
plt.savefig('cartpole-training.png')

# save training logs
# Convert tensors to dictionaries
torch.save(logs, 'cartpole-train-val-logs.pkl')

# plot coverage
# plt.figure(figsize=(10, 10))
# plt.plot(logs["coverage"])
# plt.title("Coverage")
# plt.savefig('cartpole-coverage.png')

env.close()