import json
from Experience_pool import *
from ACModel import ACModel
from stock_date import *
from Experience import Experience
import multiprocessing
import os


# f = open('trade.log', 'a')
# sys.stdout = f
# sys.stderr = f
# tf预设置


def execute_model(m, episode):
    global model
    model = m
    for i in range(glo.agent_num):
        thread_list[i].episode = episode
        thread_list[i].start()
    for i in range(glo.agent_num):
        thread_list[i].join()


def run_model():
    global model
    global thread_list
    global sys_model

    # 载入经验池
    load()
    for episode in range(glo.train_times):
        # init model and env
        """
        jobs = [multiprocessing.Process(target=init_nn, args=(i, agent_list, lock)) for i in
                range(glo.agent_num)]
        for k in jobs:
            k.start()
        for k in jobs:
            k.join()
        """
        for i in range(glo.agent_num):
            thread_list[i] = ACModel(i, model)
        for t in range(glo.train_step):
            # 多线程运行模型
            flag = False
            while len(exp_pool) <= glo.experience_pool_size:
                if len(exp_pool) < glo.experience_pool_size:
                    # 观察环境模式
                    flag = True
                    print("经验池大小：" + str(len(exp_pool)))
                execute_model('r', episode)
                """
                jobs = [multiprocessing.Process(target=run_nn, args=(
                    agent_list[i], i, episode, experience_pool, experience_cursor, lock)) for i in range(glo.agent_num)]
                for k in jobs:
                    k.start()
                for k in jobs:
                    k.join()
                """
                if not flag:
                    break
            # 如果刚刚从观察模式跳出
            if flag:
                # 保存经验池
                save_experience_pool()
            if sys_model.__contains__("train") or sys_model.__contains__("both"):
                # 执行训练
                execute_model('t', episode)
                """
                jobs = [multiprocessing.Process(target=train_nn, args=(agent_list[i], experience_pool, i, agent_list))
                        for i in
                        range(glo.agent_num)]
                for k in jobs:
                    k.start()
                for k in jobs:
                    k.join()
                """
        # 保存经验池
        if episode % glo.train_times == int(glo.train_times / 10):
            save_experience_pool()
            # 保存权重
            for i in range(glo.agent_num):
                thread_list[i].save_weights()


"""
def save_weights():
    main_actor_net.save_weights('main_actor_weights.h5', overwrite=True)
    target_actor_net.save_weights('target_actor_weights.h5', overwrite=True)
    main_critic_net.save_weights('main_critic_weights.h5', overwrite=True)
    target_critic_net.save_weights('target_critic_weights.h5', overwrite=True)
    print("权重存储完成")
"""


def save_experience_pool():
    print("经验存储中......请勿退出!!!")
    with open("Data/experience_pool.json", "w", encoding='UTF-8') as f:
        json.dump(exp_pool, f, default=lambda obj: obj.__dict__)
    print("经验存储完成")


"""
def save_model():
    main_actor_net.save('main_actor.h5', overwrite=True, include_optimizer=True)
    target_actor_net.save('target_actor.h5', overwrite=True, include_optimizer=True)
    main_critic_net.save('main_critic.h5', overwrite=True, include_optimizer=True)
    target_critic_net.save('target_critic.h5', overwrite=True, include_optimizer=True)
    print("模型存储完成")
"""
if __name__ == '__main__':
    # 全局变量
    model = multiprocessing.Value('u', "r")
    thread_list = [None for i in range(glo.agent_num)]
    sys_model = input("请输入运行模式：run\\train\\both\n")
    run_model()
