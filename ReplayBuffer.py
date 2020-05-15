import numpy as np


class ReplayBuffer(object):

    def __init__(self, max_capacity, obs_shape, act_shape):
        self.max_capacity = int(max_capacity)
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.size = 0
        self.pointer = 0
        # Memory - all arrays are created at the start to ensure that there is sufficient memory for the experiment
        self.obs = np.zeros((self.max_capacity, *self.obs_shape), dtype=np.float)
        self.new_obs = np.zeros((self.max_capacity, *self.obs_shape), dtype=np.float)
        self.rewards = np.zeros((self.max_capacity, 1), dtype=np.float)
        self.actions = np.zeros((self.max_capacity, *self.act_shape), dtype=np.float)
        self.dones = np.zeros((self.max_capacity, 1), dtype=np.bool)

    def get_experience(self, idx):
        """
        Return experiences from buffer
        :param idx: list/tuple of indices of experiences sampled
        :return: batch of sampled experiences in the form of (observations, actions, rewards, next_observations, dones)
        """
        return self.obs[idx], self.actions[idx], self.rewards[idx], self.new_obs[idx], self.dones[idx]

    def add(self, obs, act, rew, new_obs, done):
        self.obs[self.pointer] = obs
        self.actions[self.pointer] = act
        self.rewards[self.pointer] = rew
        self.new_obs[self.pointer] = new_obs
        self.dones[self.pointer] = done
        # Update size and pointer attribute
        self.pointer = (self.pointer + 1) % self.max_capacity
        self.size = max(self.pointer, min(self.size + 1, self.max_capacity))

    def get_size(self):
        return self.size
