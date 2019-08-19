import random

count = 32
frequency = '1d'
day = 5
stock_code_list = ['000517.XSHE', '000938.XSHE', '600094.XSHG', '600519.XSHG', '601318.XSHG']
# DDPG超参数
train_times = 1000
train_step = 2000
gamma = 0.4
mini_batch_size = 64
experience_pool_size = 50000
tau = 0.001
stock_state_size = 6
agent_state_size = 3
action_size = 2
epsilon = 0.1
agent_num = 2
actor_learning_rate = 0.000001
critic_learning_rate = 0.0001
# ES超参数
npop = 50
sigma = 0.1
alpha = 0.001


def random_stock_code():
    t = random.randint(0, len(stock_code_list) - 1)
    return stock_code_list[t]
