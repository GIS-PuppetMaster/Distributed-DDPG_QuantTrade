class Experience:
    stock_state = [[]]
    agent_state = []
    action = []
    reward = 0
    stock_state2 = [[]]
    agent_state2 = []

    def __init__(self, stock_state, agent_state, action, reward, stock_state2, agent_state2):
        self.stock_state = stock_state
        self.agent_state = agent_state
        self.action = action
        self.reward = reward
        self.stock_state2 = stock_state2
        self.agent_state2 = agent_state2

    def get_experience(self):
        return {'stock_state': self.stock_state, 'agent_state': self.action, 'action': self.action,
                'reward': self.reward, 'stock_state2': self.stock_state2, 'agent_state2': self.agent_state2}



    def __repr__(self):
        return repr(
            (self.stock_state, self.agent_state, self.action, self.reward, self.stock_state2, self.agent_state2))

    @staticmethod
    def object_hook(d):
        return Experience(d['stock_state'], d['agent_state'], d['action'], d['reward'], d['stock_state2'],
                          d['agent_state2'])
