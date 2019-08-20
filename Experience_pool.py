import random
import numpy as np
import glo
from multiprocessing import Lock

exp_pool = []
experience_cursor = 0
lock = Lock()


def get_info_from_experience_list(experience_list):
    """
    从经验列表中整理状态动作和奖励
    :param experience_list: 经验列表
    :return: list：股票状态、智能体状态、动作、奖励、next_股票状态、next_智能体状态
    """
    stock_res = 0
    agent_res = 0
    action_res = 0
    reward_res = 0
    stock_res2 = 0
    agent_res2 = 0
    for i in range(experience_list.__len__()):
        ex = experience_list[i]
        if i == 0:
            stock_res = ex.stock_state
            agent_res = ex.agent_state
            action_res = ex.action
            reward_res = ex.reward
            stock_res2 = ex.stock_state2
            agent_res2 = ex.agent_state2
        else:
            stock_res = np.concatenate((ex.stock_state, stock_res), axis=0)
            agent_res = np.concatenate((ex.agent_state, agent_res), axis=0)
            action_res = np.concatenate((ex.action, action_res), axis=0)
            reward_res = np.concatenate((ex.reward, reward_res), axis=0)
            stock_res2 = np.concatenate((ex.stock_state2, stock_res2), axis=0)
            agent_res2 = np.concatenate((ex.agent_state2, agent_res2), axis=0)
    return stock_res.tolist(), agent_res.tolist(), action_res.tolist(), reward_res.tolist(), stock_res2.tolist(), agent_res2.tolist()


def append_experience(experience):
    """
    向经验池中添加经验，如果经验池已满，则从index=0开始覆盖
    使用进程锁，进程安全
    :param experience: 要添加的经验
    :return: None
    """
    global experience_cursor
    global exp_pool
    global lock
    with lock:
        experience_cursor = (experience_cursor + 1) % glo.experience_pool_size
        if experience_cursor >= len(exp_pool):
            exp_pool.append(experience)
        else:
            exp_pool[experience_cursor] = experience


def get_experience_batch():
    """
    获取经验训练包
    :return: 从经验池中随机采样的经验训练包
    """
    global exp_pool
    return random.sample(exp_pool, glo.mini_batch_size)


def load():
    import os
    import json
    from  Experience import Experience
    global exp_pool
    if os.path.exists("Data/experience_pool.json"):
        print("载入经验池")
        with open("Data/experience_pool.json", "r", encoding="UTF-8") as f:
            s = f.read()
            exp_pool = json.loads(s, object_hook=Experience.object_hook)
        print("已载入经验池")
