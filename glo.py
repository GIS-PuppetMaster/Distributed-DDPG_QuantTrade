import random
import pandas as pd
import numpy as np

count = 32
# 必须为min
frequency = '30m'
day = 5
stock_code_list = ['000517.XSHE', '000938.XSHE', '600094.XSHG', '600519.XSHG', '601318.XSHG']
# DDPG超参数
train_times = 1000
train_step = 1000
gamma = 0.99
mini_batch_size = 64
experience_pool_size = 10000
tau = 0.001
stock_state_size = 6
agent_state_size = 3
action_size = 2
epsilon = 0.1
agent_num = 8
actor_learning_rate = 0.0001
critic_learning_rate = 0.00001
# 每轮训练画多少次图像
draw_frequency = 10
# 一共保存多少次经验
save_exp_frequency = 10
# ES超参数
npop = 50
sigma = 0.1
alpha = 0.001
# 只读数据，初始化后不要进行操作
data = {}
date = {}
dict = {}


def init():
    # 必须在主程序中调用一次
    print("初始化全局data/date......")
    for s in stock_code_list:
        data[s] = pd.read_csv('Data/' + s.replace(".", "_") + ".csv", index_col='Unnamed: 0')
    for s in stock_code_list:
        temp = np.array(data[s].index)
        date[s] = temp.reshape(len(temp), )
    for s in stock_code_list:
        date_list = date[s]
        temp = {}
        for i in range(len(date_list)):
            temp[date_list[i]] = i
        dict[s] = temp


def random_stock_code():
    t = random.randint(0, len(stock_code_list) - 1)
    return stock_code_list[t]
