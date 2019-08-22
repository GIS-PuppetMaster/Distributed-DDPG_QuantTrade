import json

from ACModel import ACModel
from stock_date import *
import multiprocessing
from Experience_pool import Experience_pool
import os


# f = open('trade.log', 'a')
# sys.stdout = f
# sys.stderr = f
# tf预设置


def execute_model(m):
    global model
    global thread_flag
    global time_stamp
    # 执行前清空flag
    for i in range(glo.agent_num):
        thread_flag[i] = ''
    if m == 'i':
        print("初始化模型")
        for i in range(glo.agent_num):
            # value必须倒数第二个设置
            model.value = m
            # time_stamp最后设置，表示命令已更新
        if time_stamp.value < 1000:
            time_stamp.value += 1
        else:
            time_stamp.value = 0
        for i in range(glo.agent_num):
            thread_list[i].start()
    else:
        print("执行模型操作：" + m)
        for i in range(glo.agent_num):
            model.value = m
        if time_stamp.value < 1000:
            time_stamp.value += 1
        else:
            time_stamp.value = 0

    # 等待线程完成反馈
    print("等待进程执行中..................................................")
    flag = True
    start_time = datetime.now()
    while flag:
        if (datetime.now() - start_time).seconds >= 120:
            print("threadflag:" + str(thread_flag))
            print("time_stamp:" + str(time_stamp.value))
            break
        pause_counter = 0
        flag = False
        # print(str(thread_flag))
        for i in range(glo.agent_num):
            thread = thread_flag[i]
            if thread == 'p':
                pause_counter += 1
            if thread != m and thread!='p':
                flag = True
                break
            # 全部暂停，表示全部满一年，可以进行下一轮训练
            if pause_counter == glo.agent_num:
                return True
    print("进程执行完毕")
    return False


def run_model():
    global model
    global thread_list
    global thread_flag
    global sys_model
    global time_stamp
    global multi_step
    global multi_episode
    # 建立经验池
    ep = Experience_pool()
    # 载入经验池
    ep.load()
    print("建立模型")
    # 建立模型
    for i in range(glo.agent_num):
        thread_list[i] = ACModel(i, model, thread_flag, ep, time_stamp, multi_episode, multi_step)
    execute_model('i')
    for episode in range(glo.train_times):
        multi_episode.value = episode
        # init model and env
        print("初始化环境")
        for i in range(glo.agent_num):
            thread_list[i].init_env()
        for t in range(glo.train_step):
            multi_step.value = t
            # 多进程运行模型
            print("episode:" + str(episode))
            print("     times:" + str(t))
            print("运行模型")
            flag = False
            pause = False
            while len(ep.exp_pool) <= glo.experience_pool_size:
                pause = False
                if len(ep.exp_pool) < glo.experience_pool_size:
                    # 观察环境模式
                    flag = True
                    # 连续运行
                    print("观察模式，经验池大小：" + str(len(ep.exp_pool)))
                if execute_model('r'):
                    pause = True

                if len(ep.exp_pool) == glo.experience_pool_size:
                    if flag:
                        print("观察完成,经验池大小：" + str(len(ep.exp_pool)))
                        save_experience_pool(ep)
                    break
            if pause:
                break
            if sys_model.__contains__("train") or sys_model.__contains__("both"):
                # 执行训练
                print("执行训练")
                if execute_model('t'):
                    break
        # 保存经验池
        if episode % (glo.train_times / glo.save_exp_frequency) == 0 and episode != 0:
            save_experience_pool(ep)
        # 每个episode保存权重
        execute_model('s')
    # 终止所有进程
    print("终止进程")
    execute_model('e')


def save_experience_pool(ep):
    print("经验存储中......请勿退出!!!")
    with open("Data/experience_pool.json", "w", encoding='UTF-8') as f:
        exp_list = []
        for i in range(glo.experience_pool_size):
            exp_list.append(ep.exp_pool[i])
        json.dump(exp_list, f, default=lambda obj: obj.__dict__)
    print("经验存储完成")


if __name__ == '__main__':
    # 全局变量
    model = multiprocessing.Value('u', 'i')
    multi_episode = multiprocessing.Value('L', 0)
    multi_step = multiprocessing.Value('L', 0)

    thread_flag = multiprocessing.Manager().list(range(glo.agent_num))
    thread_list = [None for i in range(glo.agent_num)]
    time_stamp = multiprocessing.Value('L', 0)
    # sys_model = input("请输入运行模式：run\\train\\both\n")
    sys_model = 'both'
    glo.init()
    run_model()
