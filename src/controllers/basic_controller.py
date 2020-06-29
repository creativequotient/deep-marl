import torch as T
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from learners import REGISTRY as learner_registry
from components.action_selector import ActionSelector
from components.replay_buffer import ReplayBuffer
from utils.common import weight_update


class BasicController(object):
    def __init__(self, obs_shape_n, act_shape_n, state_shape, act_type, args):
        self.n_agents = len(obs_shape_n)

        self.device = T.device('cuda' if args['exp']['cuda'] else 'cpu')

        self.action_selector = ActionSelector(act_shape_n, act_type, args, self.device)

        self._build_agents(obs_shape_n, act_shape_n, state_shape, args)

        self.replay_buffer = ReplayBuffer(args['learner']['buffer_size'], obs_shape_n, act_shape_n, state_shape)

        self.batch_size = args['learner']['batch_size']
        self.min_buffer_size = self.batch_size * args['env']['episode_len']

    def _build_agents(self, obs_shape_n, act_shape_n, state_shape, args):
        self.agents = []
        learner = learner_registry[args['learner']['learner_name']]
        for idx in range(self.n_agents):
            self.agents.append(learner(f'agent_{idx}', idx, act_shape_n, obs_shape_n, state_shape, args['learner'], self, self.device))

    def step(self, obs_n, mask_n, explore, target=False, provide_logits=False):
        """
        obs_n         : List of observations corresponding to each agent
        explore       : True if noise/exploration required, False otherwise
        target        : True if action is to be sampled from target network
        provide_logits: True if logits (before sampling) are to be returned, False otherwise
        """
        for idx, (obs, mask) in enumerate(zip(obs_n, mask_n)):
            if len(obs.shape) == 1:
                obs_n[idx] = obs.reshape(-1, *obs.shape)
            if len(mask.shape) == 1:
                mask_n[idx] = mask.reshape(-1, *mask.shape)

        if not target:
            logits_n = [agent.actor(obs) for agent, obs in zip(self.agents, obs_n)]
        else:
            logits_n = [agent.target_actor(obs) for agent, obs in zip(self.agents, obs_n)]

        # set logits on invalid actions to a negative number
        for idx, (logit, mask) in enumerate(zip(logits_n, mask_n)):
            mask = T.tensor(mask, device=self.device)
            neg = T.ones_like(logit) * -99999
            logits_n[idx] = T.where(mask != 0, logit, neg)

        act_n = self.action_selector(logits_n, explore)

        return (act_n, logits_n) if provide_logits else act_n

    def experience(self, **kwargs):
        self.replay_buffer.experience(**kwargs)

    def update(self):
        if self.replay_buffer.size < self.min_buffer_size:
            return

        for agent in self.agents:
            agent.prep_training()

        for agent in self.agents:
            batch = self.replay_buffer.sample(self.batch_size)
            q_loss, pi_loss = agent.update(batch)

        for agent in self.agents:
            weight_update(agent.actor, agent.target_actor, agent.polyak)
            weight_update(agent.critic, agent.target_critic, agent.polyak)
            agent.prep_roll_outs()
