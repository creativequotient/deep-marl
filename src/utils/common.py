import torch as T
import numpy as np


def weight_update(source, target, polyak):
    with T.no_grad():
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data = polyak * target_param.data + (1 - polyak) * param.data


def onehot_from_logits(logits, device):
    logits = logits.detach().cpu().numpy()
    output = np.zeros(logits.shape)
    for idx, entry in enumerate(logits):
        output[idx][np.argmax(entry)] = 1.0
    return T.tensor(output, dtype=T.float64, device=device)


def exp_to_tensors(exp_n_batch, device):
    act_n, avail_n, obs_n, next_obs_n, next_avail_n, rew_n, done_n = [], [], [], [], [], [], []
    for agent_exp in exp_n_batch:
        act_n.append(T.tensor(agent_exp['act'], dtype=T.float64, device=device))
        avail_n.append(T.tensor(agent_exp['avail'], device=device))
        obs_n.append(T.tensor(agent_exp['obs'], dtype=T.float64, device=device))
        next_obs_n.append(T.tensor(agent_exp['next_obs'], dtype=T.float64, device=device))
        next_avail_n.append(T.tensor(agent_exp['next_avail'], device=device))
        rew_n.append(T.tensor(agent_exp['rew'], dtype=T.float64, device=device))
        done_n.append(T.tensor(agent_exp['done'], dtype=T.float64, device=device))
    return act_n, avail_n, obs_n, next_obs_n, next_avail_n, rew_n, done_n
