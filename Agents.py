import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import gumbel_softmax

from RandomProcess import OrnsteinUhlenbeckActionNoise
from ReplayBuffer import ReplayBuffer
from Common import onehot_from_logits

MSELoss = nn.MSELoss()


class MADDPGAgent(object):

    def __init__(self, id, local_q, model, num_units, lr, gamma, tau, batch_size, min_replay_size, replay_buffer_size,
                 update_interval, obs_shape, act_shape, discrete, global_observation_shape, global_action_shape):

        # Metadata
        self.id = id
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.discrete = discrete

        # ReplayBuffer
        self.min_replay_size = min_replay_size
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(max_capacity=self.replay_buffer_size, obs_shape=obs_shape,
                                          act_shape=act_shape)

        # Training parameters
        self.update_interval = update_interval
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.local_q = local_q
        self.batch_size = batch_size
        self.obs_shape = obs_shape
        self.act_shape = act_shape

        # Noise params
        if not self.discrete:
            self.noise = OrnsteinUhlenbeckActionNoise(np.zeros(act_shape, dtype=np.float))
        else:
            self.noise = 0.5

        # Networks
        self.pi = model(self.lr, self.obs_shape, self.act_shape,
                        self.discrete, True, self.device,
                        num_units)
        self.pi_target = model(self.lr, self.obs_shape, self.act_shape,
                        self.discrete, True, self.device,
                        num_units)
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

        if self.local_q:
            input_shape = sum(self.obs_shape + self.act_shape)
        else:
            input_shape = sum(map(lambda x: x.shape[0], global_observation_shape)) + sum(
                map(lambda x: x.n, global_action_shape))
        self.q = model(self.lr, (input_shape,), (1,), self.discrete, False, self.device, num_units)
        self.q_target = model(self.lr, (input_shape,), (1,), self.discrete, False, self.device, num_units)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=self.lr)

        self.hard_update()

        self.prep_rollouts()

    def experience(self, obs, act, rew, new_obs, done):
        self.replay_buffer.add(obs, act, rew, new_obs, done)

    def reset_noise(self):
        if self.discrete:
            self.noise = self.noise * 0.9995
        if not self.discrete:
            self.noise.reset()

    def get_action(self, obs, noise=False):
        if type(obs) == np.ndarray:
            obs = T.tensor(obs, dtype=T.double, device=self.device).unsqueeze(0)
        with T.no_grad():
            if not self.discrete:
                action = self.pi(obs)
                if noise:
                    action += T.tensor(self.noise(), dtype=T.double, device=self.device).unsqueeze(0)
                return action.clamp(-1, 1)
            else:
                logits = self.pi(obs)
                if noise:
                    return gumbel_softmax(logits, hard=True)
                else:
                    return gumbel_softmax(logits, hard=True)

    def get_experience(self, idx):
        return self.replay_buffer.get_experience(idx)

    def soft_update(self, amount):
        with T.no_grad():
            for pi_param, target_pi_param in zip(self.pi.parameters(), self.pi_target.parameters()):
                target_pi_param.data = amount * pi_param.data + (1 - amount) * target_pi_param.data

            for q_param, target_q_param in zip(self.q.parameters(), self.q_target.parameters()):
                target_q_param.data = amount * q_param.data + (1 - amount) * target_q_param.data

    def hard_update(self):
        self.soft_update(1.0)

    def update(self, agents, i):
        assert agents[i] == self
        # Check if it is time to update
        if self.replay_buffer.get_size() < self.min_replay_size:
            return None
        # Check for local_q parameter
        if self.local_q:
            agents = [self]

        info = {}

        # Update Q function
        sampled_idx = np.random.choice(self.replay_buffer.get_size(), self.batch_size, replace=False)
        # Collate experiences
        global_obs = []
        global_actions = []
        global_new_obs = []
        for agent in agents:
            o, a, r, o_, d = list(
                map(lambda x: T.tensor(x, dtype=T.double, device=self.device), agent.get_experience(sampled_idx)))
            global_obs.append(o)
            global_actions.append(a)
            global_new_obs.append(o_)

        # Calculate target, actual Qs
        all_target_acts = [gumbel_softmax(agent.pi(obs), hard=True) if self.discrete else agent.pi(obs) for agent, obs in zip(agents, global_new_obs)]
        target_q_in = T.cat((*global_new_obs, *all_target_acts), 1)
        obs, _, rew, _, done = self.get_experience(sampled_idx)
        obs, rew, done = list(map(lambda x: T.tensor(x, dtype=T.double, device=self.device), [obs, rew, done]))
        with T.no_grad():
            target_q_value = rew + self.gamma * (1 - done) * self.q_target(target_q_in)
        # Calculate predicted Qs
        actual_q_in = T.cat((*global_obs, *global_actions), 1)
        actual_q_value = self.q(actual_q_in)
        # Compute loss
        q_loss = MSELoss(actual_q_value, target_q_value.detach())
        info['q_loss'] = q_loss
        # Update gradients
        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 0.5)
        self.q_optimizer.step()

        # Update Pi network
        if self.discrete:
            pi_logits = self.pi(obs)
            pi_act = gumbel_softmax(pi_logits, hard=True)
        else:
            pi_logits = self.pi(obs)
            pi_act = pi_logits
        all_pi_acts = []
        for agent, o in zip(agents, global_obs):
            if agent != self:
                all_pi_acts.append(gumbel_softmax(agent.pi(o), hard=True) if self.discrete else agent.pi(o))
            else:
                all_pi_acts.append(pi_act)
        q_in = T.cat((*global_obs, *all_pi_acts), 1)
        pi_loss = -self.q(q_in).mean()
        pi_loss += (pi_logits ** 2).mean() * 1e-3  # Regularization term
        info['pi_loss'] = pi_loss
        # Update gradients
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
        self.pi_optimizer.step()

        return info

    def save_agent(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        target_pi_path = os.path.join(save_path, "target_pi_network.pth")
        T.save(self.pi_target.state_dict(), target_pi_path)

        target_q_path = os.path.join(save_path, "target_q_network.pth")
        T.save(self.q_target.state_dict(), target_q_path)

        pi_path = os.path.join(save_path, "pi_network.pth")
        T.save(self.pi.state_dict(), pi_path)

        q_path = os.path.join(save_path, "q_network.pth")
        T.save(self.q.state_dict(), q_path)

    def load_agent(self, save_path):
        pi_path = os.path.join(save_path, "pi_network.pth")
        self.pi.load_state_dict(T.load(pi_path))
        self.pi.eval()

        target_pi_path = os.path.join(save_path, "target_pi_network.pth")
        self.pi_target.load_state_dict(T.load(target_pi_path))
        self.pi_target.eval()

        q_path = os.path.join(save_path, "q_network.pth")
        self.q.load_state_dict(T.load(q_path))
        self.q.eval()

        target_q_path = os.path.join(save_path, "target_q_network.pth")
        self.q_target.load_state_dict(T.load(target_q_path))
        self.q_target.eval()

        self.hard_update()

    def prep_training(self):
        self.pi.train()
        self.pi_target.train()
        self.q.train()
        self.q_target.train()

    def prep_rollouts(self):
        self.pi.eval()
        self.pi_target.eval()
        self.q.eval()
        self.q_target.eval()