import torch
from torch import nn
from tensordict.nn.distributions import NormalParamExtractor

def quantile_loss(q_low, q_high, actions, alpha):
    """
    q_low: predicted alpha quantile
    q_high: predicted (1-alpha) quantile
    actions: actual actions taken
    alpha: quantile level in (0, 1) ex. 0.1 -> 10th percentile

    q_low, q_high, actions should all have same dim

    TODO: right now this loss is not normalized (in [0,1]) because the actions
    are not *required* to be normalized to [0,1] in the action space
    """
    # assert q_low.shape == q_high.shape == actions.shape
    # print(f'q_low {q_low.shape} q_high {q_high.shape}')
    # q_low = q_low.squeeze()
    # q_high = q_high.squeeze()
    # assert torch.le(q_low, q_high)

    low_loss = torch.where(
        actions >= q_low,
        alpha * (actions - q_low),
        (1 - alpha) * (q_low - actions)
    )
    
    high_loss = torch.where(
        actions <= q_high,
        (1 - alpha) * (q_high - actions),
        alpha * (actions - q_high)
    )

    return low_loss, high_loss



class ActorWithQuantiles(nn.Module):
    """Separates policy outputs from quantile outputs"""
    def __init__(self, actor_net, action_dim, action_low, action_high, device):
        super().__init__()
        self.actor_net = actor_net  # Everything up to but NOT including extractor
        self.normal_extractor = NormalParamExtractor()
        self.action_dim = action_dim
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32, device=device))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32, device=device))
    
    def forward(self, x):
        # print('==================================================================')   
        x = self.actor_net(x)
        # print(f'input shape {x.shape}')
        # Split outputs: first 2*action_dim for policy, last 2*action_dim for quantiles
        policy_out = x[..., :2*self.action_dim]
        quantile_out = x[..., 2*self.action_dim:]
        # print(f"Policy out shape: {policy_out.shape}")
        # print(f"Quantile out shape: {quantile_out.shape}")

        # Extract loc and scale normally
        # From: https://docs.pytorch.org/rl/stable/reference/generated/torchrl.modules.tensordict_module.ProbabilisticActor.html#torchrl.modules.tensordict_module.ProbabilisticActor
        #   "loc" and "scale" for the Normal distribution and similar
        loc, scale = self.normal_extractor(policy_out)
        # print(f"After normal_extractor - loc shape: {loc.shape}, scale shape: {scale.shape}")
        
        # Extract and clamp quantiles to be within the action space
        q_low_raw = quantile_out[..., :self.action_dim]
        q_high_raw = quantile_out[..., self.action_dim:]
        # print(f"q_low_raw shape: {q_low_raw.shape}")
        # print(f"q_high_raw shape: {q_high_raw.shape}")
        
        q_low = torch.clamp(q_low_raw, self.action_low, self.action_high)
        q_high = torch.clamp(q_high_raw, self.action_low, self.action_high)
        # print(f"q_low (after clamp) shape: {q_low.shape}")
        # print(f"q_high (after clamp) shape: {q_high.shape}")
        # print('==================================================================')
        # raise Exception('debug purpose quit in ActorWithQuantiles')
        return loc, scale, q_low, q_high