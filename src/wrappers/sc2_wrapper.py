import numpy as np

from smac.env import StarCraft2Env
from .base_wrapper import BaseWrapper

class SC2Wrapper(StarCraft2Env, BaseWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env_info = self.get_env_info()

    @property
    def action_spaces(self):
        return [self.env_info['n_actions'] for _ in range(self.num_agents)]

    @property
    def observation_spaces(self):
        return [self.env_info['obs_shape'] for _ in range(self.num_agents)]


    @property
    def action_space_type(self):
        return 'discrete'

    @property
    def state_spaces(self):
        return self.env_info['state_shape']

    @property
    def num_agents(self):
        return self.env_info['n_agents']

    def get_avail_actions(self):
        # Create action space mask
        return np.array(super().get_avail_actions())

    def step(self, act_n):
        act_n = [np.argmax(act) for act in act_n] # convert 1 hot encodings to ints
        rew, done, _ = super().step(act_n)
        rew_n = [rew] * self.num_agents
        done_n = [done] * self.num_agents
        obs_n = super().get_obs()
        state = super().get_state()
        return obs_n, rew_n, done_n, state

    def reset(self):
        super().reset()
        return super().get_obs(), super().get_state()

    def info(self):
        return self.num_agents, self.observation_spaces, self.state_spaces, self.action_spaces, self.action_space_type
