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
with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    rollout = env.rollout(max_steps=1000, policy=policy_module)


# Extract results
rewards = rollout["next", "reward"]
actions = rollout["action"]
q_low = rollout.get("q_low", None)
q_high = rollout.get("q_high", None)

total_reward = rewards.sum().item()
step_count = rollout["step_count"].max().item()

print(f"Eval complete")
print(f"Total reward: {total_reward:.2f}")
print(f"Steps survived: {step_count}")

if q_low is not None and q_high is not None:
    coverage = ((actions >= q_low) & (actions <= q_high)).float().mean().item()
    print(f"Action quantile coverage: {coverage * 100:.2f}%")
else:
    print("HELP no quantile outputs found in policy rollout.")

# Plot reward per step
plt.plot(rewards.squeeze().numpy())
plt.title("Reward per Step")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid()
plt.tight_layout()
plt.savefig("eval-reward-curve.png")
print("Reward plot saved as eval-reward-curve.png")

env.close()
