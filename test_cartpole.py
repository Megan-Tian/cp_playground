from collections import defaultdict
import torch
import matplotlib.pyplot as plt
from torchrl.envs import GymEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import set_exploration_type, ExplorationType
# from torchrl.data import TensorDict
from tqdm import tqdm

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
torch.serialization.add_safe_globals([ProbabilisticActor])

def load_and_evaluate_policy(policy_path='cartpole-policy.pth', num_episodes=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the trained policy
    policy_module = torch.load(policy_path, 
                            #    map_location=torch.device('cpu'), 
                               weights_only=False)
    print(f"Loaded policy from {policy_path}")

    # Create a fresh environment
    env = GymEnv('InvertedDoublePendulum-v4', device=device)
    env = TransformedEnv(
        env,
        Compose(
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    # Initialize logging variables
    logs = defaultdict(list)
    print(logs)

    # Run evaluation (no exploration, i.e., deterministic policy)
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for i in tqdm(range(num_episodes), desc="Evaluating Policy"):
            env.reset()
            # execute a rollout with the trained policy
            # each env.rollout() creates a new tensordict to advance the states, actions for that episode
            eval_rollout = env.rollout(1000, policy_module)

            logs["eval_reward_step_avg"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval_reward_episode_total"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval_step_count"].append(eval_rollout["step_count"].max().item())
            del eval_rollout

    print(f'logs {logs}')
    return
   



if __name__ == "__main__": 
    # load and display logs
    train_val_logs = torch.load('cartpole-logs.pkl', weights_only=False)
    print(train_val_logs)
    load_and_evaluate_policy(policy_path='cartpole-policy.pth', num_episodes=10)
