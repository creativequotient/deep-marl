import numpy as np
import torch as T


class ReplayBuffer(object):

    def __init__(self, max_capacity, obs_shape_n, act_shape_n, state_shape):
        self.max_capacity = int(max_capacity)
        self.obs_shape_n = obs_shape_n
        self.act_shape_n = act_shape_n
        self.curr_size = 0
        self.pointer = 0
        # Memory - all arrays are created at the start to ensure that there is sufficient memory for the experiment
        self.agent_buffers = [{} for _ in obs_shape_n]
        self.states = np.zeros((self.max_capacity, state_shape), dtype=np.float)
        self.next_states = np.zeros((self.max_capacity, state_shape), dtype=np.float)
        for idx, storage in enumerate(self.agent_buffers):
            storage['obs'] = np.zeros((self.max_capacity, self.obs_shape_n[idx]), dtype=np.float)
            storage['avail'] = np.zeros((self.max_capacity, self.act_shape_n[idx]), dtype=np.float)
            storage['next_avail'] = np.zeros((self.max_capacity, self.act_shape_n[idx]), dtype=np.float)
            storage['next_obs'] = np.zeros((self.max_capacity, self.obs_shape_n[idx]), dtype=np.float)
            storage['rew'] = np.zeros((self.max_capacity, 1), dtype=np.float)
            storage['act'] = np.zeros((self.max_capacity, self.act_shape_n[idx]), dtype=np.float)
            storage['done'] = np.zeros((self.max_capacity, 1), dtype=np.float)

    def make_index(self, batch_size):
        return np.random.choice(self.size, batch_size, replace=False)

    def sample(self, batch_size):
        sample_idx = self.make_index(batch_size)
        result = {'agents': [{} for _ in self.agent_buffers]}
        for idx, storage in enumerate(self.agent_buffers):
            result['agents'][idx]['obs'] = storage['obs'][sample_idx]
            result['agents'][idx]['next_obs'] = storage['next_obs'][sample_idx]
            result['agents'][idx]['rew'] = storage['rew'][sample_idx]
            result['agents'][idx]['act'] = storage['act'][sample_idx]
            result['agents'][idx]['avail'] = storage['avail'][sample_idx]
            result['agents'][idx]['next_avail'] = storage['next_avail'][sample_idx]
            result['agents'][idx]['done'] = storage['done'][sample_idx]
        result['state'] = self.states[sample_idx]
        result['next_state'] = self.next_states[sample_idx]
        return result

    def experience(self, act_n, avail_n, obs_n, next_avail_n, rew_n, next_obs_n, done_n, state, next_state):
        for buffer, act, avail, obs, next_avail, rew, next_obs, done in zip(self.agent_buffers, act_n, avail_n, obs_n, next_avail_n, rew_n, next_obs_n, done_n):
            buffer['obs'][self.pointer] = obs
            buffer['avail'][self.pointer] = avail
            buffer['next_obs'][self.pointer] = next_obs
            buffer['next_avail'][self.pointer] = next_avail
            buffer['rew'][self.pointer] = rew
            buffer['act'][self.pointer] = act
            buffer['done'][self.pointer] = done
        self.states[self.pointer] = state
        self.next_states[self.pointer] = next_state
        # Update size and pointer attribute
        self.pointer = (self.pointer + 1) % self.max_capacity
        self.curr_size = max(self.pointer, min(self.size + 1, self.max_capacity))

    @property
    def size(self):
        return self.curr_size
