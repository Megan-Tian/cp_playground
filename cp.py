# https://arxiv.org/pdf/2107.07511
# 2.2 conformalized QR
# 2.3.1 conformalized point pred

import numpy as np
import torch
from torchrl.envs import Compose, DoubleToFloat, ObservationNorm, StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

# Load trained policy
policy_path = "cartpole-policy.pth"
policy_module = torch.load(policy_path, map_location=torch.device("cpu"), weights_only=False)
print(f'policy_module spec: {policy_module.spec}')
policy_module.eval()

# Setup environment (matching training env)
env = TransformedEnv(
    GymEnv("InvertedDoublePendulum-v4", device="cpu"),
    Compose(
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

# Evaluate policy
# TODO add more rollouts/evals
with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    rollout = env.rollout(max_steps=1000, policy=policy_module)

# with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
#     tensordict = env.reset()
#     done = False
#     print(f'tensordict[observation]: {tensordict["observation"]}')
#     while not done:
#         print(f'observation: {tensordict["observation"]}')
#         input_td = tensordict.select("observation")

#         input_td = policy_module(input_td)
#         print(f'input_td: {input_td}')
#         # tensordict["action"] = action
#         print(f'action mean = {input_td['loc']}, action variance = {input_td["scale"]}')
#         tensordict["action"] = input_td["action"]
#         tensordict = env.step(tensordict)
#         done = tensordict.get("done", False).item()


env.close()
# check shapes, should all have same first dimension (number of steps)
print(rollout.keys())
print(rollout["observation"].shape)
print(rollout["action"].shape)
print(rollout["loc"].shape)
print(rollout["scale"].shape)

# uncertainty: variance of each action predicted by the policy
variance = rollout["scale"] # TODO: i should rename this this key :D

# score function = uncertainty/variance
score = variance

# compute quantile of calibration scores
n = len(rollout["scale"])
alpha=0.05
q_hat = np.quantile(score, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

print(f'scores: {score[:5]}')
print(f'alpha = {alpha}')
print(f'q_hat = {q_hat}')



means = rollout['loc'][:, 0]
stds = np.sqrt(rollout['scale'][:, 0])
actions = rollout['action'][:, 0]

upper_bound = means + 2 * stds
lower_bound = means - 2 * stds

x = np.arange(len(means)) 

plt.scatter(x, means, color='blue', label='Means')
plt.scatter(x, actions, color='red', label='Actions')
plt.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.2)
plt.xlabel('Index')
plt.ylabel('Mean Value')
plt.legend()
plt.show()