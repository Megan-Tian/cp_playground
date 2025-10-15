import numpy as np
from matplotlib import pyplot as plt
import torch

def analyze():
    datapath = 'cartpole-train-val-logs.pkl'
    logs = torch.load(datapath, weights_only=False)
    
    # check sHaPEs
    print(type(logs))
    print(logs.keys())
    print(f'eval_actions shape {logs["eval_actions"][0].shape}')
    print(f'eval_q_low shape {logs["eval_q_low"][0].shape}')
    print(f'eval_q_high shape {logs["eval_q_high"][0].shape}')

    # plot eval_actions, eval_q_low, eval_q_high - check of true action bounded by predicted quantiles
    for i in range(len(logs["eval_actions"])):
        plt.plot(range(len(logs["eval_actions"][i])), logs["eval_actions"][i].squeeze(-1).cpu(), label='eval_actions')
        plt.plot(range(len(logs["eval_q_low"][i])), logs["eval_q_low"][i].squeeze(-1).cpu(), label='eval_q_low', linestyle='--')
        plt.plot(range(len(logs["eval_q_high"][i])), logs["eval_q_high"][i].squeeze(-1).cpu(), label='eval_q_high', linestyle='--')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    analyze()