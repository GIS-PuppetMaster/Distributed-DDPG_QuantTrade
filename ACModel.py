class ACModel:
    def __init__(self, actor, target_actor, critic, target_critic, env, stock_state, agent_state):
        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic
        self.env = env
        self.stock_state = stock_state
        self.agent_state = agent_state
