import socket
from dataclasses import dataclass
from datetime import datetime
import warnings

from tensordict import TensorDictParams, NestedKey

import utils
warnings.filterwarnings("ignore")
from torch import multiprocessing

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import tensordict
import torchrl
from tensordict.nn import TensorDictModule, composite_lp_aggregate
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import MultiStepTransform
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.objectives import SoftUpdate, LossModule, ValueEstimators
from torchrl.objectives.value import TD0Estimator
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torchrl.objectives.utils import _vmap_func

from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger
from torchrl.envs import TransformedEnv, DMControlEnv

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

# QR-SAC Hyperparameters
num_cells = 256
critic_lr = 5.0e-4
policy_lr = 2.5e-4
max_grad_norm = 10.0

n_step = 1

# Quantile-specific
num_quantiles = 16  # M in pseudocode
kappa = 1.0  # Huber loss threshold

# Data collection
frames_per_batch = 1024
total_frames = 15_000_000

# Training
sub_batch_size = 256
utd_ratio = 1.0

# SAC specific
gamma = 0.99
tau = 0.005
#alpha_init = 1.0
alpha_init = 0.1
target_entropy = "auto"

# Replay buffer
buffer_size = 1_000_000
init_random_frames = 5000 * n_step

#env_name = "InvertedDoublePendulum-v4"
env_name = "InvertedPendulum-v5"
#env_name = "CartPole-v1"


# ============================================
# QR-SAC Loss Module
# ============================================

class QRSACLoss(LossModule):
    """
    Quantile Regression SAC Loss Module.

    Automatically creates twin critics and target networks from a single critic definition.
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"advantage"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            state_action_value (NestedKey): The input tensordict key where the
                state action value is expected.  Defaults to ``"state_action_value"``.
            log_prob (NestedKey): The input tensordict key where the log probability is expected.
                Defaults to ``"sample_log_prob"`` when :func:`~tensordict.nn.composite_lp_aggregate` returns `True`,
                `"action_log_prob"`  otherwise.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

        action: NestedKey = "action"
        value: NestedKey = "state_value"
        state_action_value: NestedKey = "state_action_value"
        log_prob: NestedKey | None = None
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

        def __post_init__(self):
            if self.log_prob is None:
                if composite_lp_aggregate(nowarn=True):
                    self.log_prob = "sample_log_prob"
                else:
                    self.log_prob = "action_log_prob"

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys
    default_value_estimator = ValueEstimators.TD0

    actor_network: TensorDictModule
    qvalue_network: TensorDictModule
    actor_network_params: TensorDictParams
    qvalue_network_params: TensorDictParams
    value_network_params: TensorDictParams | None
    target_actor_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams
    target_value_network_params: TensorDictParams | None


    def __init__(
        self,
        actor_network,
        qvalue_network,
        num_quantiles=32,
        kappa=1.0,
        num_qvalue_nets=2,
        gamma=0.99,
        alpha_init=1.0,
        n_step=7,
        target_entropy="auto",
        delay_qvalue=True,
        deactivate_vmap=False
    ):
        super().__init__()

        self.num_quantiles = num_quantiles
        self.kappa = kappa
        self.num_qvalue_nets = num_qvalue_nets
        self.gamma = gamma
        self.delay_qvalue = delay_qvalue
        self.n_step = n_step

        # Store actor network
        self.actor_network = actor_network

        self.deactivate_vmap = deactivate_vmap

        # Use convert_to_functional to properly set up critic networks
        # This creates the twin Q-networks and target networks with proper naming
        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=self.delay_qvalue
        )

        # Entropy temperature (log_alpha for numerical stability)
        # Detect device from module parameters
        try:
            device = next(self.parameters()).device
        except (StopIteration, AttributeError):
            device = getattr(torch, "get_default_device", lambda: torch.device("cpu"))()

        self.log_alpha = torch.nn.Parameter(
            torch.tensor([alpha_init], dtype=torch.float32, device=device).log()
        )
        self.target_entropy = target_entropy

        self._vmap_qnetworkN0 = _vmap_func(
            self.qvalue_network,
            (None, 0),
            randomness=self.vmap_randomness,
            pseudo_vmap=self.deactivate_vmap,
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def actor_network_params(self):
        """Return actor parameters for optimizer"""
        return self.actor_network


    def make_value_estimator(self, value_type=None, **hyperparams):
        """
        Stub for TorchRL API compatibility.

        QR-SAC computes quantile targets directly in forward(), so this is a no-op.
        The gamma parameter is already handled in __init__.
        """
        if value_type is not None or hyperparams:
            import warnings
            warnings.warn(
                "QRSACLoss.make_value_estimator() is a no-op. "
                "Quantile targets are computed directly in forward(). "
                "Use gamma parameter in __init__ instead.",
                UserWarning
            )

    def _quantile_huber_loss(self, quantiles, target_quantiles):
        """Compute quantile regression loss with Huber loss"""


        # Reshape for pairwise comparison
        y = target_quantiles.unsqueeze(2)  # (batch, M, 1)
        x = quantiles.unsqueeze(1)  # (batch, 1, M)
        diff = y - x  # (batch, M, M)

        # Huber loss
        abs_diff = torch.abs(diff)
        huber_loss = torch.where(
            abs_diff < self.kappa,
            0.5 * diff ** 2,
            self.kappa * (abs_diff - 0.5 * self.kappa)
        )

        # Compute quantile fractions (τ̂)
        steps = torch.arange(self.num_quantiles, dtype=torch.float32, device=quantiles.device)
        tau_hat = ((steps + 1) / self.num_quantiles + steps / self.num_quantiles) / 2.0
        tau_hat = tau_hat.view(1, 1, self.num_quantiles)  # (1, 1, M)

        # Quantile regression weight
        with torch.no_grad():
            delta = (diff < 0).float()
        weight = torch.abs(tau_hat - delta)

        # Element-wise quantile loss
        element_wise_loss = weight * huber_loss  # (batch, M, M)

        # Sum over predicted quantiles (j), mean over target quantiles (i)
        loss = element_wise_loss.sum(dim=2).mean(dim=1)  # (batch,)

        return loss

    def forward(self, tensordict_in):
        """
        Compute all three losses: actor, critic, and alpha.

        Args:
            tensordict_in: Batch of transitions with keys:
                - "observation": current state
                - "action": taken action
                - ("next", "observation"): next state
                - ("next", "reward"): reward
                - ("next", "done"): terminal flag

        Returns:
            TensorDict with loss values
        """
        obs = tensordict_in["observation"]
        action = tensordict_in["action"]
        next_obs = tensordict_in["next", "observation"]
        reward = tensordict_in["next", "reward"]
        done = tensordict_in["next", "done"]

        # Ensure alpha is on the same device as the data
        alpha = self.alpha.to(obs.device).detach()

        # ============================================
        # Compute next actions (no gradients)
        # ============================================
        with torch.no_grad():
            next_state_dict = tensordict.TensorDict(
                {"observation": next_obs}, batch_size=next_obs.shape[0]
            )
            next_action_dist = self.actor_network(next_state_dict)
            next_action = next_action_dist["action"]
            next_log_prob = next_action_dist["action_log_prob"].unsqueeze(-1)

        # ============================================
        # POLICY LOSS
        # ============================================
        current_state_dict = tensordict.TensorDict(
            {"observation": obs}, batch_size=obs.shape[0]
        )
        action_dist = self.actor_network(current_state_dict)
        sampled_action = action_dist["action"]
        log_prob = action_dist["action_log_prob"].unsqueeze(-1)

        # Get quantiles from current critics for sampled actions
        sampled_state_action = tensordict.TensorDict({
            "observation": obs,
            "action": sampled_action
        }, batch_size=obs.shape[0])

        # Evaluate current critics on sampled actions
        sampled_q_quantiles = self._vmap_qnetworkN0(
            sampled_state_action, self.qvalue_network_params
        )

        # Compute mean Q-values for each network: shape (2, batch_size, M)
        sampled_q_mean = sampled_q_quantiles["quantiles"].mean(dim=2, keepdim=True)

        # Take minimum over the two critics: shape (batch_size, 1)
        q_min = sampled_q_mean.min(dim=0).values

        loss_actor = (alpha * log_prob - q_min).mean()

        # ============================================
        # CRITIC LOSS
        # ============================================
        with torch.no_grad():
            # Get next quantiles from target networks
            next_state_action = tensordict.TensorDict({
                "observation": next_obs,
                "action": next_action
            }, batch_size=next_obs.shape[0])

            next_tensordict_expand = self._vmap_qnetworkN0(
                next_state_action, self.target_qvalue_network_params
            )

            # Get all quantiles: shape (2, batch_size, M)
            quantiles_all = next_tensordict_expand["quantiles"]

            # Select minimum mean Q-value network
            q_next_mean = quantiles_all.mean(dim=2, keepdim=True)  # (2, batch_size, 1)

            # We take the lowest q-estimate (along network dimension)
            q_to_use = q_next_mean.argmin(dim=0)  # (batch_size, 1)

            # Select quantiles corresponding to the minimum mean Q-value
            # Expand q_to_use to match quantiles shape for gather
            idx = q_to_use.expand(-1, self.num_quantiles)  # (B, M)
            q_to_use_expanded = idx.unsqueeze(0)  # (1, B, M)
            quantiles_next = torch.gather(quantiles_all, 0, q_to_use_expanded).squeeze(0)  # (batch_size, M)

            entropy = alpha * next_log_prob
            # FIXME: Does the multistep transform already apply the discount?
            target_quantiles = reward + (self.gamma ** self.n_step) * (1 - done.float()) * (quantiles_next - entropy) # (batch_size, M)

        current_state_action = tensordict.TensorDict({
            "observation": obs,
            "action": action
        }, batch_size=obs.shape[0])

        current_q_expand = self._vmap_qnetworkN0(
            current_state_action, self.qvalue_network_params
        )
        # Predict current quantiles (observed, on buffer actions)
        q1_pred = current_q_expand[0]["quantiles"]
        q2_pred = current_q_expand[1]["quantiles"]

        # Compute losses
        critic1_loss = self._quantile_huber_loss(q1_pred, target_quantiles).mean()
        critic2_loss = self._quantile_huber_loss(q2_pred, target_quantiles).mean()

        # Use MSE loss as alternative to debug
        #critic1_loss = 0.5 * ((q1_pred - target_quantiles) ** 2).mean()
        #critic2_loss = 0.5 * ((q2_pred - target_quantiles) ** 2).mean()
        loss_qvalue = critic1_loss + critic2_loss

        # ============================================
        # ALPHA LOSS (entropy temperature)
        # ============================================
        if self.target_entropy == "auto":
            # Automatic entropy target (negative action dimension)
            target_entropy_value = -sampled_action.shape[-1]
        else:
            target_entropy_value = self.target_entropy

        loss_alpha = -(self.log_alpha * (log_prob + target_entropy_value).detach()).mean()

        # Return losses in TensorDict format with diagnostics
        return tensordict.TensorDict({
            "loss_actor": loss_actor,
            "loss_qvalue": loss_qvalue,
            "loss_alpha": loss_alpha,
            "alpha": alpha,
            "entropy": -log_prob.mean(),
            # Diagnostic values
            "q_pred_mean": q1_pred.mean(),
            "q_pred_std": q1_pred.std(),
            "q_pred_min": q1_pred.min(),
            "q_pred_max": q1_pred.max(),
            "q_pred_spread": (q1_pred.max(1).values - q1_pred.min(1).values).mean(),  # Difference between highest and lowest quantile
            "q_policy_mean": q_min.mean(),
            "target_mean": target_quantiles.mean(),
            "target_std": target_quantiles.std(),
            "target_min": target_quantiles.min(),
            "target_max": target_quantiles.max(),
            "target_spread": (target_quantiles[:, -1] - target_quantiles[:, 0]).mean(),
            "td_error": (target_quantiles - q1_pred).abs().mean(),
            "reward_mean": reward.mean(),
        }, batch_size=[])


def make_env(from_pixels=False):
    """Create environment with standard transforms"""
    base_env = GymEnv(
        env_name,
        device=device,
        from_pixels=from_pixels,
        pixels_only=False if from_pixels else True,
    )

    env = TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    return env


def make_qr_sac_agent(env):
    """Create QR-SAC actor and quantile critic networks"""

    ################ ACTOR (Policy) ################
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
        actor_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"]
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
        default_interaction_type=ExplorationType.RANDOM,
    )

    ################ QUANTILE CRITIC ################
    class QuantileNet(nn.Module):
        """Quantile network that outputs M quantile values"""
        def __init__(self, num_quantiles):
            super().__init__()
            self.num_quantiles = num_quantiles
            self.network = nn.Sequential(
                nn.LazyLinear(num_cells, device=device),
                nn.Tanh(),
                nn.LazyLinear(num_cells, device=device),
                nn.Tanh(),
                nn.LazyLinear(num_cells, device=device),
                nn.Tanh(),
                nn.LazyLinear(num_quantiles, device=device),  # Output M quantiles
            )

        def forward(self, observation, action):
            # Concatenate observation and action
            x = torch.cat([observation, action], dim=-1)
            raw_outputs = self.network(x)  # Shape: (batch, M)
            # We don't need to sort here; the pairwise loss handles it
            return raw_outputs

    qvalue_net = TensorDictModule(
        QuantileNet(num_quantiles),
        in_keys=["observation", "action"],
        out_keys=["quantiles"],
    )

    return policy_module, qvalue_net


def quantile_huber_loss(quantiles, target_quantiles, kappa=1.0):
    """
    Compute quantile regression loss with Huber loss.

    Args:
        quantiles: Predicted quantiles, shape (batch_size, num_quantiles)
        target_quantiles: Target quantiles, shape (batch_size, num_quantiles)
        kappa: Huber loss threshold

    Returns:
        Loss tensor of shape (batch_size,)
    """
    batch_size = quantiles.shape[0]
    num_quantiles = quantiles.shape[1]

    # Reshape for pairwise comparison
    # y[b, i, 1] - x[b, 1, j] = diff[b, i, j]
    y = target_quantiles.unsqueeze(2)  # (batch, M, 1)
    x = quantiles.unsqueeze(1)  # (batch, 1, M)

    diff = y - x  # (batch, M, M)

    # Huber loss
    abs_diff = torch.abs(diff)
    huber_loss = torch.where(
        abs_diff < kappa,
        0.5 * diff ** 2,
        kappa * (abs_diff - 0.5 * kappa)
    )

    # Compute quantile fractions (τ̂)
    steps = torch.arange(num_quantiles, dtype=torch.float32, device=quantiles.device)
    tau_hat = ((steps + 1) / num_quantiles + steps / num_quantiles) / 2.0
    tau_hat = tau_hat.view(1, 1, num_quantiles)  # (1, 1, M)

    # Quantile regression weight
    delta = (diff < 0).float()  # Indicator function
    weight = torch.abs(tau_hat - delta)

    # Element-wise quantile loss
    element_wise_loss = weight * huber_loss  # (batch, M, M)

    # Sum over predicted quantiles (j), mean over target quantiles (i)
    loss = element_wise_loss.sum(dim=2).mean(dim=1)  # (batch,)

    return loss


def train():
    env = make_env(from_pixels=False)
    check_env_specs(env)

    # Create QR-SAC agent
    policy_module, qvalue_net = make_qr_sac_agent(env)

    # Test policy
    test_td = env.reset()
    test_td_with_action = policy_module(test_td)
    print("Running policy:", test_td_with_action)

    # Test quantile network
    test_quantiles = qvalue_net(test_td_with_action)
    print("Running quantile network:", test_quantiles)
    print("Quantile shape:", test_quantiles["quantiles"].shape)

    ################ QR-SAC LOSS MODULE ################
    loss_module = QRSACLoss(
        actor_network=policy_module,
        qvalue_network=qvalue_net,
        num_quantiles=num_quantiles,
        kappa=kappa,
        num_qvalue_nets=2,
        n_step=n_step,
        gamma=gamma,
        alpha_init=alpha_init,
        target_entropy=target_entropy,
        delay_qvalue=True,
    )

    # Target network updater
    target_net_updater = SoftUpdate(loss_module, tau=tau)

    ################ REPLAY BUFFER ################
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=buffer_size, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=sub_batch_size,
        transform=MultiStepTransform(n_steps=n_step, gamma=gamma),
    )

    ################ DATA COLLECTOR ################
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
        init_random_frames=init_random_frames,
    )

    ################ OPTIMIZERS ################
    actor_optim = torch.optim.Adam(loss_module.actor_network_params.parameters(), lr=policy_lr)
    critic_optim = torch.optim.Adam(loss_module.qvalue_network_params.parameters(), lr=critic_lr)
    alpha_optim = torch.optim.Adam([loss_module.log_alpha], lr=policy_lr)

    ###################### TRAIN LOOP ######################
    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = current_time + "_" + socket.gethostname()
    writer = SummaryWriter(f"runs/{env_name}_qr_sac_{total_frames}/{log_dir}")

    collected_frames = 0

    for i, tensordict_data in enumerate(collector):
        # Add collected data to replay buffer
        tensordict_data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(tensordict_data_view)
        collected_frames += tensordict_data.numel()

        # Only train after initial random exploration
        if collected_frames >= init_random_frames:
            num_updates = int(frames_per_batch * utd_ratio)

            for update_idx in range(num_updates):
                # Sample from replay buffer
                subdata = replay_buffer.sample(sub_batch_size)

                # Compute all losses
                loss_vals = loss_module(subdata)

                # Actor loss
                actor_loss = loss_vals["loss_actor"]
                actor_optim.zero_grad()
                actor_loss.backward()
                # gt-sophy doesn't clip the actor gradients
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.actor_network_params.parameters(), max_grad_norm
                )
                actor_optim.step()

                # Critic loss
                critic_loss = loss_vals["loss_qvalue"]
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.qvalue_network_params.parameters(), max_grad_norm
                )
                critic_optim.step()

                # Alpha (entropy temperature) loss
                alpha_loss = loss_vals["loss_alpha"]
                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()

                # Update target networks
                target_net_updater.step()

                # Logging
                if update_idx == 0:
                    writer.add_scalar("loss/train_actor", actor_loss.item(), collected_frames)
                    writer.add_scalar("loss/train_critic", critic_loss.item(), collected_frames)
                    writer.add_scalar("loss/train_alpha_loss", alpha_loss.item(), collected_frames)
                    writer.add_scalar("loss/train_alpha_value", loss_vals["alpha"].item(), collected_frames)
                    writer.add_scalar("loss/train_entropy", loss_vals["entropy"].item(), collected_frames)
                    writer.add_scalar("loss/train_total", actor_loss.item() + critic_loss.item() + alpha_loss.item(), collected_frames)

                    # Diagnostic logging
                    writer.add_scalar("debug/q_pred_mean", loss_vals["q_pred_mean"].item(), collected_frames)
                    writer.add_scalar("debug/q_pred_std", loss_vals["q_pred_std"].item(), collected_frames)
                    writer.add_scalar("debug/q_pred_min", loss_vals["q_pred_min"].item(), collected_frames)
                    writer.add_scalar("debug/q_pred_max", loss_vals["q_pred_max"].item(), collected_frames)
                    writer.add_scalar("debug/q_policy_mean", loss_vals["q_policy_mean"].item(), collected_frames)
                    writer.add_scalar("debug/target_mean", loss_vals["target_mean"].item(), collected_frames)
                    writer.add_scalar("debug/target_std", loss_vals["target_std"].item(), collected_frames)
                    writer.add_scalar("debug/target_min", loss_vals["target_min"].item(), collected_frames)
                    writer.add_scalar("debug/target_max", loss_vals["target_max"].item(), collected_frames)
                    writer.add_scalar("debug/q_pred_spread", loss_vals["q_pred_spread"].item(), collected_frames)
                    writer.add_scalar("debug/target_spread", loss_vals["target_spread"].item(), collected_frames)
                    writer.add_scalar("debug/td_error", loss_vals["td_error"].item(), collected_frames)
                    writer.add_scalar("debug/reward_mean", loss_vals["reward_mean"].item(), collected_frames)

                    writer.add_scalar("debug/critic_grad_norm", critic_grad_norm.item(), collected_frames)
                    writer.add_scalar("debug/actor_grad_norm", actor_grad_norm.item(), collected_frames)

        # Log collection statistics
        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"avg reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(actor_optim.param_groups[0]["lr"])

        writer.add_scalar("train/reward_avg", logs["reward"][-1], collected_frames)
        writer.add_scalar("train/step_count_max", logs["step_count"][-1], collected_frames)
        writer.add_scalar("train/buffer_size", len(replay_buffer), collected_frames)

        # Evaluation
        if i % 10 == 0 and collected_frames >= init_random_frames:
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

                writer.add_scalar("eval/reward_cumulative", logs['eval reward (sum)'][-1], collected_frames)
                writer.add_scalar("eval/reward_avg", logs["eval reward"][-1], collected_frames)
                writer.add_scalar("eval/max_step_count", logs["eval step_count"][-1], collected_frames)

                del eval_rollout

        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str]))

    # Save policy and logs
    torch.save(policy_module, f'baseline/{env_name}-qr-sac-policy-{total_frames}.pth')
    print(f'Saved policy!')
    torch.save(logs, f'baseline/{env_name}-qr-sac-train-val-logs-{total_frames}.pkl')

    writer.flush()
    writer.close()


def eval():
    ################## LOAD POLICY ##################
    policy_path = f"baseline/{env_name}-qr-sac-policy-{total_frames}.pth"
    print(f'Loading policy from {policy_path}')
    policy_module = torch.load(policy_path, map_location=torch.device("cpu"), weights_only=False)
    policy_module.eval()

    ################## SETUP ENV ##################
    env = make_env(from_pixels=True)
    policy_module(env.reset())
    print("Running policy:", policy_module(env.reset()))
    check_env_specs(env)

    ################## ROLLOUT (manual stepping) ##################
    max_ep_steps = 1000
    rollout = tensordict.TensorDict(batch_size=[max_ep_steps])

    _data = env.reset()
    i = 0
    done = False

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        while i < max_ep_steps and not done:
            action_td = policy_module(_data.to_tensordict())
            _data["action"] = action_td.get("action")
            _data = env.step(_data)
            rollout[i] = _data.clone()
            _data = torchrl.envs.utils.step_mdp(_data, keep_other=True)
            done = _data["terminated"].item() or _data["truncated"].item() or _data["done"].item()
            i += 1

    print(f'Rollout finished at step {i}')
    rollout = rollout[:i]

    utils.video_writer(rollout["pixels"], f"{env_name}_videos/{env_name}-qr-sac-{datetime.now()}.mp4")

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

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(rewards.squeeze().cpu().numpy(), label="reward")
    plt.title(f"Reward per Step (Total: {total_reward:.2f})")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    action_np = actions.detach().cpu().numpy().squeeze()
    steps = range(len(action_np))
    plt.plot(steps, action_np, label="action (deterministic)", marker="o", markersize=2)

    obs_np = observations.cpu().numpy()
    for j in range(min(4, obs_np.shape[1])):
        plt.plot(steps, obs_np[:, j], label=f"observation[{j}]", alpha=0.5)

    plt.title("Actions and Observations")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    env.close()


if __name__ == "__main__":
    train()
    eval()