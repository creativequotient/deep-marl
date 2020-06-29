from multiagent.environment import MultiAgentEnv
from multiagent import scenarios
from wrappers.base_wrapper import BaseWrapper
from gym import spaces
import numpy as np

class MPEWrapper(MultiAgentEnv, BaseWrapper):
    def __init__(self, world, reset, reward, obs):
        super().__init__(world, reset, reward, obs)

    @staticmethod
    def make_mpe_env(scenario_name, n_adv):
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        env = MPEWrapper(world, scenario.reset_world, scenario.reward, scenario.observation)
        return env

    @property
    def action_spaces(self):
        result = []

        for idx, space in enumerate(self.action_space):
            result.append(space.n)

        if self.action_space_type == 'multi_discrete':
            raise NotImplementedError

        return result

    @property
    def observation_spaces(self):
        obs_shapes = []
        for idx, space in enumerate(self.observation_space):
            obs_shapes.append(space.shape[0])
        return obs_shapes

    @property
    def action_space_type(self):
        if isinstance(self.action_space[0], spaces.discrete.Discrete):
            return 'discrete'
        elif isinstance(self.action_space[0], spaces.multi_discrete.MultiDiscrete):
            return 'multi-discrete'
        else:
            return 'continuous'

    @property
    def state_spaces(self):
        return 0

    @property
    def num_agents(self):
        return len(self.observation_space)

    def get_avail_actions(self):
        # Create action space mask
        result = []
        for space in self.action_spaces:
            result.append(np.array([1 for i in range(space)]))

        return result

    def step(self, act_n):
        obs_n, rew_n, done_n, _ = super().step(act_n)
        return obs_n, rew_n, done_n, []

    def reset(self):
        return super().reset(), []
