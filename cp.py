# https://arxiv.org/pdf/2107.07511
# 2.2 conformalized QR
# 2.3.1 conformalized point pred

import numpy as np
import torch
from torchrl.envs import Compose, DoubleToFloat, ObservationNorm, StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
import matplotlib.pyplot as plt
from cartpole_rf_qr3 import *

np.random.seed(0)
torch.manual_seed(0)

def main():
    # get data from rollouts
    all_actions, all_q_lows, all_q_highs, all_scales = eval(3)
    all_actions = np.concatenate(all_actions, axis=0)
    all_q_lows = np.concatenate(all_q_lows, axis=0)
    all_q_highs = np.concatenate(all_q_highs, axis=0)
    all_scales = np.concatenate(all_scales, axis=0).squeeze()
    
    print(f'all_actions shape: {all_actions.shape} | all_q_lows shape: {all_q_lows.shape} | all_q_highs shape: {all_q_highs.shape} | all_scales shape: {all_scales.shape}')

    # compute scores
    scores = np.maximum(all_q_highs - all_actions, all_actions - all_q_lows)
    print(f'scores shape: {scores.shape}')
    
    # compute desired quantile of scores
    alpha = 0.2 # alpha/2 = q_low | q_high = 1 - alpha/2
    q_hat = np.quantile(scores, np.ceil((scores.shape[0] + 1) * (1-alpha)) / scores.shape[0])
    print(f'q_hat score threshold for alpha {alpha}: {q_hat}')
    
    # shrink intervals
    def compute_conformalized_quantiles(q_low, q_high, q_hat=q_hat):
        return q_low - q_hat, q_high + q_hat
    
    return q_hat


if __name__ == "__main__":
    q_hat = main()
    eval(3, conformalized=True, q_hat=q_hat)