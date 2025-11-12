import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from torch.distributions import Normal
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from tqdm import tqdm

# An attempt at implementing Distributional Soft Actor-Critic (DSAC)
# Paper: https://arxiv.org/pdf/2004.14547
# Official implementation: https://github.com/xtma/dsac/tree/master
#
# This implementation has some parts simplified / ommitted for clarity. Major differences include:
# - No risk-sensitive policy updates
# - No soft updates for the policy network (only for the 2 target Q-nets)
# - Quantile network
#       - IQN does not use cosine embeddings
#       - Implemented fixed and IQN quantile generation, no FQF
# - Replay buffer is simple FIFO with uniform sampling (SAC has all sorts of replay buffer tricks to help 
# stabilize training... engineering details?)
#
# Questions and potential bugs / confusion are marked with TODO comments.
#
#
# Note about runtime: training for 300 episodes of 500 steps each takes ~15min locally on a laptop RTX 4070 gpu


class ReplayBuffer:
    """Simple experience replay buffer. FIFO with uniform sampling."""
    def __init__(self, capacity: int = 100_000):
        # oldest experiences automatically removed when capacity exceeded
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)


class QuantileNetwork(nn.Module):
    """
    Quantile value network Z(s, a, tau)
    
    Note this is not the same as reference implementation which uses IQN with cosine embeddings (appendix B eq 18)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 num_quantiles: int = 32):
        super().__init__()
        self.num_quantiles = num_quantiles
        
        # input is concatenated [state, action, tau]
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim), # +1 is for tau (scalar)
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action, tau):
        """
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            tau: (batch_size, num_quantiles)
            
        Returns:
            quantile_values: (batch_size, num_quantiles)
        """
        batch_size = state.shape[0]
        num_quantiles = tau.shape[1]
        
        # expand state and action to match quantiles
        state_expanded = state.unsqueeze(1).expand(-1, num_quantiles, -1)  # (B, num_quantiles, state_dim)
        action_expanded = action.unsqueeze(1).expand(-1, num_quantiles, -1)  # (B, num_quantiles, action_dim)
        tau_expanded = tau.unsqueeze(-1)  # (B, num_quantiles, 1)
        
        # concatenate all inputs to match input shape to self.net
        inputs = torch.cat([state_expanded, action_expanded, tau_expanded], dim=-1)  # (B, num_quantiles, state_dim + action_dim + 1)
        inputs_flat = inputs.reshape(-1, inputs.shape[-1])  # (B * num_quantiles, state_dim + action_dim + 1)
        
        # send through network
        outputs_flat = self.net(inputs_flat)  # (B * num_quantiles, 1)
        
        # reshape back
        quantile_values = outputs_flat.reshape(batch_size, num_quantiles)  # (B, num_quantiles)
        
        return quantile_values


class GaussianPolicy(nn.Module):
    """Stochastic Gaussian policy with Tanh action squashing"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # FOR TANH ACTION SQUASHING / SCALING 
        self.action_scale = 3.0 # FOR INVERTED PENDULUM ACTIONS ARE IN [-3, 3]
        self.action_bias = 0.0
        
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # mean and std of the Gaussian the actions are implementated as separate output heads/layers on top of self.net
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state, deterministic: bool = False, return_log_prob: bool = True):
        """
        Args:
            state: (batch_size, state_dim)
        Returns:
            action, log_prob (optional)
        """
        features = self.net(state)
        
        # mean and log_std are two separate heads sharing backbone self.net
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        # TODO to clamp or not to clamp? clamped in paper impl: https://github.com/xtma/dsac/blob/master/rlkit/torch/dsac/policies.py
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max) 
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            dist = Normal(mean, std)
            # reparameterization trick
            x = dist.rsample()
            y = torch.tanh(x)
            action = y * self.action_scale + self.action_bias
            
            if return_log_prob:
                # compute log probability with tanh correction
                # see appendix C of SAC paper (?)
                # also: https://dosssman.github.io/posts/2022-04-13-sac/#policy-network-1
                log_prob = dist.log_prob(x)
                log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6) # add small epsilon > 0 for numerical stability
                log_prob = log_prob.sum(dim=-1, keepdim=True)
                mean = torch.tanh(mean) * self.action_scale + self.action_bias
            else:
                log_prob = None
        
        return action, log_prob, mean, log_std


def quantile_huber_loss(quantile_pred, target, tau, kappa: float = 1.0):
    """
    Quantile Huber loss
    
    Args:
        quantile_pred: (batch_size, num_quantiles)
        target: (batch_size, num_quantiles)
        tau: (batch_size, num_quantiles)
        kappa: Huber loss threshold
    """
    # expand dims for pairwise TD errors
    quantile_pred = quantile_pred.unsqueeze(-1)  # (B, T, 1)
    target = target.unsqueeze(1)  # (B, 1, T)
    tau = tau.unsqueeze(-1)  # (B, T, 1)
    
    errors = target - quantile_pred  # (B, T, T)
    
    # huber loss
    huber_loss = torch.where(
        errors.abs() <= kappa,
        0.5 * errors.pow(2),
        kappa * (errors.abs() - 0.5 * kappa)
    )
    
    # qr loss, (tau - td_errors < 0) does the weighting / decides what threshold to use
    quantile_weight = torch.abs(tau - (errors < 0).float())
    loss = (quantile_weight * huber_loss).mean()
    
    return loss



def quantile_regression_loss(input, target, tau, weight):
    """
    input: (N, T)
    target: (N, T)
    tau: (N, T)
    weight: (N, T)
    
    ngl this is straight up from the ref code. wrapping my head around
    how/why this is different from `quantile_huber_loss()` above...
    
    https://github.com/xtma/dsac/blob/master/rlkit/torch/td4/td4.py#L78
    """
    input = input.unsqueeze(-1)
    target = target.detach().unsqueeze(-2)
    tau = tau.detach().unsqueeze(-1)
    weight = weight.detach().unsqueeze(-2)
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    L = F.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
    sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
    rho = torch.abs(tau - sign) * L * weight
    return rho.sum(dim=-1).mean()



class DSAC:
    """Distributional Soft Actor-Critic"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_quantiles: int = 32,
        discount: float = 0.99,
        alpha: float = 0.2,
        tau: float = 0.005,
        policy_lr: float = 3e-4,
        q_lr: float = 3e-4,
        device: str = 'cpu'
    ):
        # params
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.discount = discount
        self.alpha = alpha
        self.tau = tau
        self.device = device
        
        # policy and (double) critic networks
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.zf1 = QuantileNetwork(state_dim, action_dim, hidden_dim, num_quantiles).to(device)
        self.zf2 = QuantileNetwork(state_dim, action_dim, hidden_dim, num_quantiles).to(device)
        
        # target networks (recall these track the current self.zf1, self.zf2 with small delay to stabilize updates)
        self.target_zf1 = QuantileNetwork(state_dim, action_dim, hidden_dim, num_quantiles).to(device)
        self.target_zf2 = QuantileNetwork(state_dim, action_dim, hidden_dim, num_quantiles).to(device)
        self.target_zf1.load_state_dict(self.zf1.state_dict())
        self.target_zf2.load_state_dict(self.zf2.state_dict())
        
        # optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.zf1_optimizer = optim.Adam(self.zf1.parameters(), lr=q_lr)
        self.zf2_optimizer = optim.Adam(self.zf2.parameters(), lr=q_lr)
    
    def get_tau(self, batch_size: int, mode: str = 'iqn'):
        """
        Generate `self.num_quantiles` quantile fractions for distributional RL.
        
        Args:
            batch_size: number of samples
            mode: 'fix' for evenly-spaced quantiles (0.1, 0.2, 0.3, ...)
                  'iqn' for random quantiles
        
        Returns:
            tau: (batch_size, num_quantiles) - cumulative probabilities [0, 1]
                 Example: [0.05, 0.15, 0.28, 0.45, ..., 0.98]
                 
            tau_hat: (batch_size, num_quantiles) - midpoints between quantiles
                     Used for computing quantile values Z(s, a, tau_hat)
                     Example: [0.025, 0.10, 0.215, 0.365, ..., 0.99]
                     
            presum_tau: (batch_size, num_quantiles) - widths of each quantile bin
                        Used as weights when computing E[Z] = Q
                        Example: [0.05, 0.10, 0.13, 0.17, ..., 0.02]
                        (these sum to 1.0 for each batch)
        
        """
        if mode == 'fix': # fix uniform quantiles: [1/N, 2/N, 3/N, ..., N/N]
            presum_tau = torch.ones(batch_size, self.num_quantiles, device=self.device) / self.num_quantiles
        elif mode == 'iqn': # randomly sample quantiles
            # random widths that sum to 1
            presum_tau = torch.rand(batch_size, self.num_quantiles, device=self.device) + 0.01 # add a small epsilon > 0 to avoid "quantile 0" 
            presum_tau = presum_tau / presum_tau.sum(dim=-1, keepdim=True)
        else:
            raise ValueError("Invalid mode for get_tau(). Choose 'fix' or 'iqn'")
        
        # tau values (cumulative probabilities)
        tau = torch.cumsum(presum_tau, dim=1)
        
        # compute midpoints (tau_hat) - these are what we actually feed to the network (see DSAC paper section 3.2)
        # midpoint of [tau[i-1], tau[i]] is (tau[i-1] + tau[i])/2
        tau_hat = torch.zeros_like(tau)
        tau_hat[:, 0] = tau[:, 0] / 2.0
        tau_hat[:, 1:] = (tau[:, :-1] + tau[:, 1:]) / 2.0
        
        return tau, tau_hat, presum_tau
    
    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> Dict:
        if len(replay_buffer) < batch_size:
            return {}
        
        # sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # UPDATE Q NETS
        with torch.no_grad():
            # sample next actions from current policy
            next_actions, next_log_probs, _, _ = self.policy(next_states)
            
            # get quantile fractions 
            # reference impl shares the same taus between Q-nets: https://github.com/xtma/dsac/blob/master/rlkit/torch/dsac/dsac.py#L181
            # have a set of taus for target networks and a DIFFERENT set for the critic Q nets
            next_tau, next_tau_hat, next_presum_tau = self.get_tau(batch_size)
            
            # compute target quantile values (remember to do both q nets)
            target_z1 = self.target_zf1(next_states, next_actions, next_tau_hat)
            target_z2 = self.target_zf2(next_states, next_actions, next_tau_hat)
            target_z = torch.min(target_z1, target_z2) # in algorithm 1 this is y_i in the nested for loop (min value estimate across all quantiles)
            
            # entropy bonus (sac)
            target_z = target_z - self.alpha * next_log_probs
            
            # delta^k_{ij} in algorithm 1. exclude done states in discount because those have no future rewards
            z_target = rewards + (1 - dones) * self.discount * target_z
        
        # current quantile values
        tau, tau_hat, presum_tau = self.get_tau(batch_size)
        z1_pred = self.zf1(states, actions, tau_hat)
        z2_pred = self.zf2(states, actions, tau_hat)
        
        # quantile regression losses 
        # zf1_loss = quantile_huber_loss(z1_pred, z_target, tau_hat)
        # zf2_loss = quantile_huber_loss(z2_pred, z_target, tau_hat)
        
        # TODO: weight by presum_tau in ref impl - why? 
        zf1_loss = quantile_regression_loss(z1_pred, z_target, tau_hat, presum_tau)
        zf2_loss = quantile_regression_loss(z2_pred, z_target, tau_hat, presum_tau)
        
        # update Q networks
        self.zf1_optimizer.zero_grad()
        zf1_loss.backward()
        self.zf1_optimizer.step()
        
        self.zf2_optimizer.zero_grad()
        zf2_loss.backward()
        self.zf2_optimizer.step()
        
        # UPDATE POLICY
        # new actions with reparameterized/updated samples
        new_actions, log_probs, policy_mean, policy_log_std = self.policy(states)
        
        # get q vals from quantile networks
        # TODO it is unclear to me why we get_tau() 3 times in train_step though it shows up in the ref impl
        # TODO did i do this wrong the algorithm box "says generate quantile fractions" at the top and that's it...
        new_tau, new_tau_hat, new_presum_tau = self.get_tau(batch_size)
        z1_new = self.zf1(states, new_actions, new_tau_hat)
        z2_new = self.zf2(states, new_actions, new_tau_hat)
        
        # compute expected Q values (expectation over quantiles)
        # this is for a "neutral" (aka no) risk measure - ref code and paper have more stuff wrt the policy update
        q1_new = (new_presum_tau * z1_new).sum(dim=1, keepdim=True)
        q2_new = (new_presum_tau * z2_new).sum(dim=1, keepdim=True)
        q_new = torch.min(q1_new, q2_new)
        
        # policy loss (maximize Q - alpha * entropy)
        # entropy term: (alpha * log_probs) (higher log_prob = lower entropy)
        # want to maximize Q and entropy, so minimize (alpha * log_probs - Q)
        entropy_term = (self.alpha * log_probs).mean()
        q_term = q_new.mean()
        policy_loss = entropy_term - q_term
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # UPDATE TARGET NETWORKS (SOFT UPDATE)
        # TODO: maybe don't update every step? 
        # upon further examination the ref code also does soft update on the policy network (not every step)
        self.soft_update(self.zf1, self.target_zf1)
        self.soft_update(self.zf2, self.target_zf2)
        
        return { # log like all the loss components
            'loss/zf1_loss': zf1_loss.item(),
            'loss/zf2_loss': zf2_loss.item(),
            'loss/critic_loss': (zf1_loss.item() + zf2_loss.item()) / 2,
            
            'loss/policy_loss': policy_loss.item(),
            'loss/policy_loss_entropy_term': entropy_term.item(),
            'loss/policy_loss_q_term': q_term.item(),
            
            'loss/total_loss': zf1_loss.item() + zf2_loss.item() + policy_loss.item(),
            
            'train/q_value_mean': q_new.mean().item(),
            'train/q_value_std': q_new.std().item(),
            
            'train/q1_value': q1_new.mean().item(),
            'train/q2_value': q2_new.mean().item(),
            'train/log_prob_mean': log_probs.mean().item(),
            'train/log_prob_std': log_probs.std().item(),
            
            'train/policy_mean': policy_mean.mean().item(),
            'train/policy_std': policy_log_std.exp().mean().item(),
            
            'train/z1_pred_mean': z1_pred.mean().item(),
            'train/z2_pred_mean': z2_pred.mean().item(),
            'train/z_target_mean': z_target.mean().item(),
        }
    
    def soft_update(self, source, target):
        """
        Soft update target network
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def select_action(self, state, deterministic: bool = False):
        """
        Select action from policy
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _, _ = self.policy(state, deterministic=deterministic)
        return action.cpu().numpy()[0]

    def select_action_with_quantiles(self, state, deterministic: bool = False):
        """
        Select action from policy along with `num_quantiles` unformly distributed quantile values.
        
        Returns:
            action: (action_dimm,)
            quantile_values: (num_quantiles,)
            
        Note that the number of quantiles returned is `self.num_quantiles`, which is set at agent initialization
        and trained.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _, _, _ = self.policy(state, deterministic=deterministic)
            tau, tau_hat, _ = self.get_tau(1, mode='fix')
            quantile_values = torch.min(self.zf1(state, action, tau_hat), self.zf2(state, action, tau_hat))
            
        return action.cpu().numpy()[0], quantile_values.cpu().numpy()[0]
    

def evaluate_agent(agent: DSAC, env_name: str, num_episodes: int = 10, get_quantile_preds: bool = False, max_steps_per_episode: int = 500) -> Dict:
    """
    Evaluate the agent on `num_episodes` validation episodes
    """
    env = gym.make(env_name) # make fresh env
    episode_rewards = [] # (num_episodes,)
    episode_lengths = [] # (num_episodes,)
    episode_actions = [] # (num_episodes, episode_length, 1)
    episode_quantile_values = [] # (num_episodes, episode_length, num_quantiles)
    episode_rewards_after_each_step = [] # (num_episodes, episode_length)
    
    for _ in tqdm(range(num_episodes)):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        actions = []
        action_quantiles = []
        reward_after_step = []
        
        while not done and episode_length < max_steps_per_episode:
            # action = agent.select_action(state, deterministic=True)
            if get_quantile_preds:
                action, quantile_values = agent.select_action_with_quantiles(state, deterministic=True)
                action_quantiles.append(quantile_values)
            else:
                action = agent.select_action(state, deterministic=True)
                
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            actions.append(action)
            reward_after_step.append(episode_reward)
            
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_actions.append(actions)
        episode_quantile_values.append(action_quantiles)
        episode_rewards_after_each_step.append(reward_after_step)
    
    env.close()
    
    return {
        # log these to tensorboard (scalars)
        'eval/episode_reward_mean': np.mean(episode_rewards),
        'eval/episode_reward_max': np.max(episode_rewards),
        'eval/episode_reward_min': np.min(episode_rewards),
        'eval/episode_reward_std': np.std(episode_rewards),
        'eval/episode_length_mean': np.mean(episode_lengths),
        'eval/episode_length_max': np.max(episode_lengths),
        
        # use these for plotting along steps in a single episode
        'actions': episode_actions,
        'action_quantile_preds': episode_quantile_values if get_quantile_preds else None,
        'episode_rewards': episode_rewards, # total rewards per episode
        'episode_rewards_after_each_step': episode_rewards_after_each_step  # total rewards accumulated up to each step in the episode
    }


def train_dsac(
    env_name: str = "InvertedPendulum-v4",
    max_episodes: int = 300,
    max_steps: int = 1000,
    init_random_frames: int = 1000,
    batch_size: int = 256,
    eval_interval: int = 10,
    eval_episodes: int = 5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    log_dir: str = None
):
    """Train DSAC on Gymnasium environment with TensorBoard logging"""
    
    # Create log directory
    if log_dir is None:
        timestamp = datetime.now().strftime("%m_%d_%Y_%H%M%S")
        log_dir = f"runs/dsac_{env_name}/{timestamp}"
    
    writer = SummaryWriter(log_dir)
    print(f"Logging to: {log_dir}")
    
    # create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Environment: {env_name} | State dim: {state_dim}, Action dim: {action_dim} | Device: {device}")
    
    # agent
    agent = DSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        num_quantiles=32,
        discount=0.99,
        alpha=0.2,
        device=device,
    )
    
    # replay buffer
    replay_buffer = ReplayBuffer(capacity=100000)
    
    # training tracking
    global_step = 0
    episode_rewards = []
    episode_lengths = []
    current_episode_rewards = []
    current_episode_lengths = []
    
    # TRAIN LOOP    
    for episode in tqdm(range(max_episodes)):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # select action (to add experiences to replay buffer)
            if len(replay_buffer) < init_random_frames: # initial random exploration - TODO make this a param
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False) # TODO should this be deterministic=True during training?
            
            # step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # store transition to replay buffer
            replay_buffer.push(state, action, reward, next_state, float(done))
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            global_step += 1
            
            # train agent
            if len(replay_buffer) >= batch_size:
                train_info = agent.train_step(replay_buffer, batch_size)
                
                # log training metrics
                if train_info:
                    for key, value in train_info.items():
                        writer.add_scalar(key, value, global_step)
            
            if done:
                break
        
        # episode stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        current_episode_rewards.append(episode_reward)
        current_episode_lengths.append(episode_length)
        
        # log per-episode stats
        writer.add_scalar('train/episode_reward', episode_reward, episode)
        writer.add_scalar('train/episode_length', episode_length, episode)
        writer.add_scalar('train/buffer_size', len(replay_buffer), episode)
        
        # log batch stats (max/avg over recent episodes)
        if len(current_episode_rewards) >= 10:
            writer.add_scalar('train_batch/episode_batch_reward_mean', 
                            np.mean(current_episode_rewards), episode)
            writer.add_scalar('train_batch/episode_batch_reward_max', 
                            np.max(current_episode_rewards), episode)
            writer.add_scalar('train_batch/episode_batch_length_mean', 
                            np.mean(current_episode_lengths), episode)
            writer.add_scalar('train_batch/episode_batch_length_max', 
                            np.max(current_episode_lengths), episode)
            
            # Reset batch tracking
            current_episode_rewards = []
            current_episode_lengths = []
        
        # VALIDATION
        if (episode + 1) % eval_interval == 0:
            eval_stats = evaluate_agent(agent, env_name, num_episodes=eval_episodes)
            
            # log eval metrics to tensorboard
            for key, value in eval_stats.items():
                if key.startswith('eval/'):
                    writer.add_scalar(key, value, episode)
            
            # console prints too 
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{max_episodes}")
            print(f"\tTrain | avg reward (over last 10 episodes): {avg_reward:.2f}, last episode reward: {episode_reward:.2f}")
            print(f"\tEval | avg reward: {eval_stats['eval/episode_reward_mean']:.2f}, "
                  f"max reward: {eval_stats['eval/episode_reward_max']:.2f}")
    
    env.close()
    writer.close()
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths, label='Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f'{log_dir}/training_plots.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Training plot saved to '{plot_path}'")
    
    return agent, episode_rewards


if __name__ == "__main__":
    # Train the agent
    print("=" * 60)
    print("DSAC Training - Distributional Soft Actor-Critic")
    print("=" * 60)
    
    env_name = "InvertedPendulum-v4"    
    basepath = f'dsac/{env_name}/{datetime.now().strftime("%m_%d_%Y_%H%M%S")}'
    savepath = f'{basepath}/agent.pth'
    
    agent, rewards = train_dsac(
        env_name=env_name,
        max_episodes=300,
        max_steps=500,
        batch_size=256,
        eval_interval=10,  # Evaluate every 10 episodes
        eval_episodes=5,     # Run 5 episodes per evaluation
        log_dir=basepath,
    )
    
    torch.save(agent, savepath)
    print(f'Saved agent at {basepath}/agent.pth!')

    # savepath = 'dsac/InvertedPendulum-v4/11_12_2025_002404/agent.pth'
    print(f'Loading agent from {savepath}')
    agent = torch.load(
        savepath, 
        map_location='cuda' if torch.cuda.is_available() else 'cpu', 
        weights_only=False
    )
    
    # Final test
    print("\n" + "=" * 60)
    print("Final Evaluation (10 episodes)")
    print("=" * 60)
    
    final_eval = evaluate_agent(agent, "InvertedPendulum-v4", num_episodes=5, get_quantile_preds=True)
    
    print(f"Average Reward: {final_eval['eval/episode_reward_mean']:.2f} +/- "
          f"{final_eval['eval/episode_reward_std']:.2f}")
    print(f"Max Reward: {final_eval['eval/episode_reward_max']:.2f}")
    print(f"Average Episode Length: {final_eval['eval/episode_length_mean']:.1f}")
    
    actions = np.array(final_eval['actions'])
    quantile_preds = np.array(final_eval['action_quantile_preds'])
    episode_rewards = np.expand_dims(np.array(final_eval['episode_rewards_after_each_step']), axis=-1)
    print(f'Quantile Predictions: {quantile_preds.shape}')
    print(f'Actions Taken: {actions.shape}')
    print(f'Episode Rewards: {episode_rewards.shape}')
    # print(f'episode_rewards_after_each_step: {final_eval["episode_rewards_after_each_step"]}')
    
    
    num_episodes = quantile_preds.shape[0]

    for ep in range(num_episodes):
        plt.figure(figsize=(12, 5))
        plt.title(f"Discounted rewards, predicted critic/Q-value quantiles over time for episode {ep + 1}")
        
        # Plot each of the 32 quantile lines (THESE ARE VALUES)
        for q in range(quantile_preds.shape[2]):
            plt.plot(quantile_preds[ep][:, q], color='blue', alpha=0.2, linewidth=1, label=f'Critic Quantile Pred {q}')
        
        # plot true episode reward
        # TODO this should not be the same as the quantile_preds, should instead plot reward-to-go for apples to 
        # apples comparison i think
        
        # Suppose rewards is your cumulative rewards array
        rewards = np.array(episode_rewards[ep])

        # recover individual rewards
        r = np.empty_like(rewards)
        r[0] = rewards[0]
        r[1:] = rewards[1:] - rewards[:-1]

        # compute discounted reward-to-go
        reward_to_go = np.zeros_like(r, dtype=float)
        running_sum = 0.0
        for t in reversed(range(len(r))):
            running_sum = r[t] + 0.99 * running_sum
            reward_to_go[t] = running_sum
        
        plt.plot(reward_to_go, 
                 color='green', linestyle='--', linewidth=2, label='Discounted Reward-To-Go')
        
        # Plot the actual action
        # plt.plot(actions[ep][:, 0], color='red', linewidth=2, label='Action Taken')
        
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='xx-small') # Legend outside
        plt.tight_layout()
        plt.show()

    
    print("Evaluation complete.")