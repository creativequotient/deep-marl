class BaseWrapper(object):
    def __init__(self):
        pass

    @property
    def action_spaces(self):
        print('hello')
        raise NotImplementedError

    @property
    def observation_spaces(self):
        raise NotImplementedError

    @property
    def state_spaces(self):
        raise NotImplementedError

    @property
    def num_agents(self):
        raise NotImplementedError

    def info(self):
        return self.num_agents, self.observation_spaces, self.state_spaces, self.action_spaces, self.action_space_type
