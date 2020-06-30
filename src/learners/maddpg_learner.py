import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import gumbel_softmax
import copy
import logging

from agents.actors.mlp_actor import MLP_Actor
from agents.critics.mlp_critic import MLP_Critic
from utils.common import weight_update, exp_to_tensors

MSELoss = nn.MSELoss()
# T.autograd.set_detect_anomaly(True)

class MADDPG_Learner(object):
    def __init__(self, agent_id, agent_idx, act_shape_n, obs_shape_n, state_shape, args, controller, device, **kwargs):
        # Metadata
        self.agent_id = agent_id
        self.agent_idx = agent_idx
        self.act_shape_n = act_shape_n
        self.obs_shape_n = obs_shape_n
        self.state_shape = state_shape
        self.device = device

        # Training params
        self.discount_factor = args['discount_factor']
        self.polyak = args['polyak']
        self.lr = args['learning_rate']
        self.local_q = args['local_q']
        self.num_units = args['hidden_units']
        self.obs_shape = self.obs_shape_n[self.agent_idx]
        self.act_shape = self.act_shape_n[self.agent_idx]
        self.controller = controller

        # Networks
        self.actor = MLP_Actor(self.obs_shape,
                               self.act_shape,
                               device=self.device,
                               norm_in=args['actor_bnorm'],
                               fc1_dim=self.num_units,
                               fc2_dim=self.num_units,
                               weight_init=args['weight_init'])
        self.target_actor = copy.deepcopy(self.actor)

        if self.local_q:
            q_in_shape = self.state_shape + self.act_shape + self.obs_shape
        else:
            q_in_shape = self.state_shape + sum(self.act_shape_n) + sum(self.obs_shape_n)
        self.critic = MLP_Critic(q_in_shape,
                                 1,
                                 device=self.device,
                                 norm_in=args['critic_bnorm'],
                                 fc1_dim=self.num_units,
                                 fc2_dim=self.num_units,
                                 weight_init=args['weight_init'])
        self.target_critic = copy.deepcopy(self.critic)

        # Optimisers
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimiser = optim.Adam(self.critic.parameters(), lr=self.lr)

        # Initialization
        self.initialize_networks()
        self.prep_roll_outs()

        # Misc
        self.logger = logging.getLogger(self.agent_id)

    def update(self, batch):
        # coerce batch into tensors
        act_n, avail_n, obs_n, next_obs_n, next_avail_n, rew_n, done_n = exp_to_tensors(batch['agents'], self.device)

        rew, done = rew_n[self.agent_idx], done_n[self.agent_idx]

        states = T.tensor(batch['state'], dtype=T.float64, device=self.device)
        next_states = T.tensor(batch['next_state'], dtype=T.float64, device=self.device)

        # update critic network (Q-network)
        target_act_n = self.controller.step(next_obs_n, avail_n, explore=True, target=True)

        if not self.local_q:
            target_q_in = T.cat([next_states, *next_obs_n, *target_act_n], dim=-1)
            q_in = T.cat([states, *obs_n, *act_n], dim=-1)
        else:
            target_q_in = T.cat([next_states, next_obs_n[self.agent_idx], *target_act_n[self.agent_idx]], dim=-1)
            q_in = T.cat([states, obs_n[self.agent_idx], act_n[self.agent_idx]], dim=-1)

        with T.no_grad():
            target_q = rew.view(-1,1) + self.discount_factor * (1 - done).view(-1,1) * self.target_critic(target_q_in)

        actual_q = self.critic(q_in)

        q_loss = MSELoss(actual_q, target_q.detach())
        self.critic_optimiser.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimiser.step()

        # update actor network (policy-network)
        act_n, logits_n = self.controller.step(obs_n, next_avail_n, explore=True, provide_logits=True)
        if not self.local_q:
            q_in = T.cat([states, *obs_n, *act_n], dim=-1)
        else:
            q_in = T.cat([states, obs_n[self.agent_idx], act_n[self.agent_idx]], dim=-1)
        pi_loss = -self.critic(q_in).mean() + (logits_n[self.agent_idx] ** 2).mean() * 1e-3
        self.actor_optimiser.zero_grad()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimiser.step()

        return q_loss, pi_loss

    def prep_roll_outs(self):
        self.actor.eval()
        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()

    def prep_training(self):
        self.actor.train()
        self.target_actor.train()
        self.critic.train()
        self.target_critic.train()

    def initialize_networks(self):
        self.actor.to(dtype=T.float64, device=self.device)
        self.target_actor.to(dtype=T.float64, device=self.device)
        self.critic.to(dtype=T.float64, device=self.device)
        self.target_critic.to(dtype=T.float64, device=self.device)
