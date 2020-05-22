import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import gumbel_softmax

from deepmarl.common.random_process import OrnsteinUhlenbeckActionNoise
from deepmarl.common.replay_buffer import ReplayBuffer
from deepmarl.common.distributions import onehot_from_logits

MSELoss = nn.MSELoss()


class MADDPGAgent(object):

    def __init__(self, agent_name, agent_idx, model, obs_shape_n, act_shape_n, args, local_q):

        # Metadata
        self.agent_name = agent_name
        self.agent_idx = agent_idx
        self.device = T.device(
            'cuda' if T.cuda.is_available() else 'cpu')
        self.discrete = args['discrete']

        # Training parameters
        self.update_interval = args['update_interval']
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.lr = args['lr']
        self.local_q = local_q
        self.batch_size = args['batch_size']
        self.num_units = args['num_units']
        self.obs_shape = obs_shape_n[agent_idx].shape
        self.act_shape = (act_shape_n[agent_idx].n,)  # only applies to discrete action space so far

        # ReplayBuffer
        self.min_replay_size = args['batch_size'] * args['max_episode_len']
        self.replay_buffer_size = args['buffer_size']
        self.replay_buffer = ReplayBuffer(max_capacity=self.replay_buffer_size, obs_shape=self.obs_shape,
                                          act_shape=self.act_shape)

        # Noise params
        if not self.discrete:
            self.noise = OrnsteinUhlenbeckActionNoise(np.zeros(self.act_shape, dtype=np.float))
        else:
            # Noise is handled by sampling using the gumbel-softmax trick (ST), hence, no noise parameter for
            # discrete action spaces. Might add option for epsilon-greedy exploration in future.
            self.noise = None

        # Networks
        self.pi = model(self.lr, self.obs_shape, self.act_shape,
                        self.discrete, True, self.device,
                        self.num_units)
        self.pi_target = model(self.lr, self.obs_shape, self.act_shape,
                               self.discrete, True, self.device,
                               self.num_units)
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

        if self.local_q:
            input_shape = sum(self.obs_shape + self.act_shape)
        else:
            input_shape = sum(map(lambda x: x.shape[0], obs_shape_n)) + sum(
                map(lambda x: x.n, act_shape_n))
        self.q = model(self.lr, (input_shape,), (1,), self.discrete, False, self.device, self.num_units)
        self.q_target = model(self.lr, (input_shape,), (1,), self.discrete, False, self.device, self.num_units)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=self.lr)

        self.hard_update()  # Ensure that both target/non-target networks share the same initial parameters

        self.prep_roll_outs()  # Puts model in training mode by default (for Dropout, BatchNorm, etc. layers)

    def experience(self, obs, act, rew, new_obs, done):
        self.replay_buffer.add(obs, act, rew, new_obs, done)

    def reset_noise(self):
        if self.discrete:
            pass
        else:
            self.noise.reset()

    def get_action(self, obs, explore=False):
        if type(obs) == np.ndarray:
            obs = T.tensor(obs[None], dtype=T.double, device=self.device)
        with T.no_grad():
            out = self.pi(obs)
            if not self.discrete:
                if explore:
                    out += T.tensor(self.noise(), dtype=T.double, device=self.device).unsqueeze(0)
                return out.clamp(-1, 1)[0].cpu().numpy()
            else:
                out = gumbel_softmax(out, hard=True) if explore else onehot_from_logits(out)
                return out[0].cpu().numpy()

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

        # Update Q function
        sampled_idx = self.replay_buffer.make_index(self.batch_size)
        # Collate experiences
        global_obs = []
        global_actions = []
        global_new_obs = []
        for agent in agents:
            o, a, r, o_, d = list(
                map(lambda x: T.tensor(x, dtype=T.double, device=self.device),
                    agent.replay_buffer.sample_index(sampled_idx)))
            global_obs.append(o)
            global_actions.append(a)
            global_new_obs.append(o_)

        # Calculate target, actual Qs
        all_target_acts = [gumbel_softmax(agent.pi(obs), hard=True) if self.discrete else agent.pi(obs) for agent, obs
                           in zip(agents, global_new_obs)]
        target_q_in = T.cat((*global_new_obs, *all_target_acts), 1)
        obs, _, rew, _, done = self.replay_buffer.sample_index(sampled_idx)
        obs, rew, done = list(map(lambda x: T.tensor(x, dtype=T.double, device=self.device), [obs, rew, done]))
        with T.no_grad():
            target_q_value = rew + self.gamma * (1 - done) * self.q_target(target_q_in)
        # Calculate predicted Qs
        actual_q_in = T.cat((*global_obs, *global_actions), 1)
        actual_q_value = self.q(actual_q_in)
        # Compute loss
        q_loss = MSELoss(actual_q_value, target_q_value.detach())
        # Update gradients
        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 0.5)
        self.q_optimizer.step()

        # Update Pi network
        pi_logits = self.pi(obs)
        pi_act = gumbel_softmax(pi_logits, hard=True) if self.discrete else pi_logits
        all_pi_acts = []
        for agent, o in zip(agents, global_obs):
            if agent != self:
                all_pi_acts.append(gumbel_softmax(agent.pi(o), hard=True) if self.discrete else agent.pi(o))
            else:
                all_pi_acts.append(pi_act)
        q_in = T.cat((*global_obs, *all_pi_acts), 1)
        pi_loss = -self.q(q_in).mean()
        pi_loss += (pi_logits ** 2).mean() * 1e-3  # Regularization term
        # Update gradients
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
        self.pi_optimizer.step()

        return q_loss.cpu().detach().numpy(), pi_loss.cpu().detach().numpy()

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

    def prep_roll_outs(self):
        self.pi.eval()
        self.pi_target.eval()
        self.q.eval()
        self.q_target.eval()
