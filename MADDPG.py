import json
from Experience_pool import *
from ACModel import ACModel
from stock_date import *
import multiprocessing
from Experience_pool import Experience_pool
import os


# f = open('trade.log', 'a')
# sys.stdout = f
# sys.stderr = f
# tf预设置


def execute_model(m, episode):
    global model
    global thread_flag
    if m == 'i':
        print("初始化模型")
        for i in range(glo.agent_num):
            thread_list[i].episode = episode
            # value必须最后设置
            model.value = m
            # thread_list[i].mode = model
            thread_list[i].start()
    else:
        print("执行模型操作：" + m)
        for i in range(glo.agent_num):
            thread_list[i].episode = episode
            model.value = m
            # thread_list[i].mode = model
    # 等待线程完成反馈
    print("等待进程执行中..................................................")
    flag = True
    while flag:
        flag = False
        # print(str(thread_flag))
        for i in range(glo.agent_num):
            thread = thread_flag[i]
            if thread != m:
                flag = True
                break
    print("进程执行完毕")


def run_model():
    global model
    global thread_list
    global thread_flag
    global sys_model
    # 建立经验池
    ep = Experience_pool()
    # 载入经验池
    ep.load()
    print("建立模型")
    # 建立模型
    for i in range(glo.agent_num):
        thread_list[i] = ACModel(i, model, thread_flag,ep)
    for episode in range(glo.train_times):
        # init model and env
        print("episode:" + str(episode))
        """
        jobs = [multiprocessing.Process(target=init_nn, args=(i, agent_list, lock)) for i in
                range(glo.agent_num)]
        for k in jobs:
            k.start()
        for k in jobs:
            k.join()
        """
        print("初始化环境")
        for i in range(glo.agent_num):
            thread_list[i].init_env()
        execute_model('i', episode)
        for t in range(glo.train_step):
            # 多进程运行模型
            print("times:" + str(t))
            print("运行模型")
            flag = False
            while len(ep.exp_pool) <= glo.experience_pool_size:
                if len(ep.exp_pool) < glo.experience_pool_size:
                    # 观察环境模式
                    flag = True
                    print("观察模式，经验池大小：" + str(len(ep.exp_pool)))
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
                    print("观察完成")
                    break
            # 如果刚刚从观察模式跳出
            if flag:
                # 保存经验池
                save_experience_pool()
            if sys_model.__contains__("train") or sys_model.__contains__("both"):
                # 执行训练
                print("执行训练")
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
            save_experience_pool(ep)
            # 保存权重
            for i in range(glo.agent_num):
                thread_list[i].save_weights()
    # 终止所有进程
    execute_model('e', glo.train_times)


"""
def save_weights():
    main_actor_net.save_weights('main_actor_weights.h5', overwrite=True)
    target_actor_net.save_weights('target_actor_weights.h5', overwrite=True)
    main_critic_net.save_weights('main_critic_weights.h5', overwrite=True)
    target_critic_net.save_weights('target_critic_weights.h5', overwrite=True)
    print("权重存储完成")
"""


def save_experience_pool(ep):
    print("经验存储中......请勿退出!!!")
    with open("Data/experience_pool.json", "w", encoding='UTF-8') as f:
        json.dump(ep.exp_pool, f, default=lambda obj: obj.__dict__)
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
    thread_flag = multiprocessing.Manager().list(range(glo.agent_num))
    thread_list = [None for i in range(glo.agent_num)]
    # sys_model = input("请输入运行模式：run\\train\\both\n")
    sys_model = 'both'
    run_model()
