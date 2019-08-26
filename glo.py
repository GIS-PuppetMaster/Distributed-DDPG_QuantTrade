import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import *
import json
import os

count = 32
"""
必须为min（分钟）!!!!!!!
"""
frequency = '30' + 'm'
""""""
day = 5
# , '000938.XSHE', '600094.XSHG', '600519.XSHG', '601318.XSHG'
stock_code_list = ['000517.XSHE']
# DDPG超参数
train_times = 1000
train_step = 500
gamma = 0.99
mini_batch_size = 64
experience_pool_size = 50000
tau = 0.001
stock_state_size = 6
agent_state_size = 3
action_size = 2
epsilon = 0.1
agent_num = 8
actor_learning_rate = 0.0001
critic_learning_rate = 0.001
# 每轮训练画多少次图像
draw_frequency = 10
# 一共保存多少次经验
save_exp_frequency = 10
reset_trigger = 10
reset = 20
# ES超参数
npop = 50
sigma = 0.1
alpha = 0.001
# 只读数据，初始化后不要进行操作
data = {}
date = {}
dict = {}
scaler = {}


def init():
    global data
    global date
    global dict
    all_arg = {}
    # 必须在主程序中调用一次
    print("初始化全局data/date......")
    for s in stock_code_list:
        temp_data = pd.read_csv('Data/' + s.replace(".", "_") + ".csv", index_col='Unnamed: 0')
        data[s] = temp_data
        temp_data = temp_data.values
        scale = StandardScaler().fit(temp_data)
        scaler[s] = scale
        arg = {}
        arg['mean'] = scale.mean_.tolist()
        arg['var'] = scale.var_.tolist()
        all_arg[s] = arg
    with open("Data/scaling_arg.json", "w", encoding='UTF-8') as f:
        json.dump(all_arg, f)
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
