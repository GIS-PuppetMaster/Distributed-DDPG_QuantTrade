import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import *
import json
import os
from stockstats import *
import stockstats

count = 32
"""
必须为min（分钟）!!!!!!!
"""
frequency = '30' + 'm'
""""""
day = 30
# , '000938.XSHE', '600094.XSHG', '600519.XSHG', '601318.XSHG'
stock_code_list = ['000517.XSHE']
# stock_code_list = ['601318.XSHG']
# DDPG超参数
train_times = 1000
train_step = 500
gamma = 0.99
mini_batch_size = 64
experience_pool_size = 20000
low_rate = 2
mid_rate = 3
high_rate = 5
sum_rate = low_rate + mid_rate + high_rate
tau = 0.001
stock_state_size = 19
agent_state_size = 3
price_state_size = 6
action_size = 2
epsilon = 0.1
agent_num = 8
actor_learning_rate = 0.05
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
day_data = {}
date = {}
dict = {}
scaler = {}
min_scaler = {}


def init():
    global data
    global date
    global dict
    global day_data
    day_arg = {}
    min_arg = {}
    # 必须在主程序中调用一次
    print("初始化全局data/date......")
    for s in stock_code_list:
        temp_data = pd.read_csv('Data/' + s.replace(".", "_") + ".csv", index_col='Unnamed: 0')
        state_data = pd.read_csv('Data/' + s.replace(".", "_") + "day.csv", index_col='Unnamed: 0')
        state_data = state_data.rename(columns={"money": "amount"})
        state_data = stockstats.StockDataFrame.retype(state_data)
        state_data = pd.concat([state_data[['open', 'close', 'high', 'low', 'volume', 'amount']],
                                state_data[
                                    ['close_5_sma', 'macd', 'macds', 'macdh', 'rsi_6', 'rsi_12', 'cci', 'tr', 'atr',
                                     'kdjk', 'kdjd', 'kdjj', 'wr_6']]], axis=1, sort=False)
        # 记录数据
        data[s] = temp_data
        day_data[s] = state_data
        # 记录day_data标准化参数
        state_data = state_data.values
        scale = StandardScaler().fit(state_data)
        scaler[s] = scale
        arg = {'mean': scale.mean_.tolist(), 'var': scale.var_.tolist()}
        day_arg[s] = arg
        # 记录min_data白哦准话参数
        temp_data = temp_data.values
        scale = StandardScaler().fit(temp_data)
        min_scaler[s] = scale
        arg = {'mean': scale.mean_.tolist(), 'var': scale.var_.tolist()}
        min_arg[s] = arg
    with open("Data/scaling_day_arg.json", "w", encoding='UTF-8') as f:
        json.dump(day_arg, f)
    with open("Data/scaling_min_arg.json", "w", encoding='UTF-8') as f:
        json.dump(min_arg, f)
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
