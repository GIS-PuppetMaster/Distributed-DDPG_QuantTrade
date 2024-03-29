import json
from ACModel import ACModel
from StockDate import *
import multiprocessing
from Experience_pool import Experience_pool
import os
import time
from pynput.keyboard import Key, Controller
from pynput import keyboard

# f = open('trade.log', 'a')
# sys.stdout = f
# sys.stderr = f
# tf预设置


def execute_model(m):
    global model
    global thread_flag
    global time_stamp
    global obs
    # 执行前清空flag
    for i in range(glo.agent_num):
        thread_flag[i] = ''
    if m == 'i':
        print("初始化模型")
        for i in range(glo.agent_num):
            # value必须倒数第二个设置
            model.value = m
            # time_stamp最后设置，表示命令已更新
        time.sleep(0.1)
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
        time.sleep(0.1)
        if time_stamp.value < 1000:
            time_stamp.value += 1
        else:
            time_stamp.value = 0

    # 等待线程完成反馈
    print("等待进程执行中..................................................")
    flag = True
    start_time = datetime.now()
    while flag:
        """
        if (datetime.now() - start_time).seconds >= 120 and m != 's' and m!='i':
            print("thread_flag:" + str(thread_flag))
            print("time_stamp:" + str(time_stamp.value))
            print("紧急保存权重")
            m = 's'
            for i in range(glo.agent_num):
                model.value = m
            time.sleep(0.1)
            if time_stamp.value < 1000:
                time_stamp.value += 1
            else:
                time_stamp.value = 0
            for thread in thread_list:
                thread.terminate()
            os._exit(1)
        """
        flag = False
        # print(str(thread_flag))
        pause_counter = 0
        for i in range(glo.agent_num):
            thread = thread_flag[i]
            if thread == 'p':
                pause_counter += 1
            if pause_counter == glo.agent_num:
                return True
            if thread != m and thread != 'p':
                # 未完成，继续循环
                flag = True
                break
    print("进程执行完毕")
    return False


def train_model():
    global model
    global thread_list
    global thread_flag
    global sys_model
    global time_stamp
    global multi_step
    global multi_episode
    global flag_lock
    # 建立经验池
    ep_dict = {'low': Experience_pool('low'), 'mid': Experience_pool('mid'), 'high': Experience_pool('high')}
    # 载入经验池
    for ep in list(ep_dict.values()):
        ep.load()
    print("建立模型")
    # 建立模型
    for i in range(glo.agent_num):
        thread_list[i] = ACModel(i, model, thread_flag, ep_dict, time_stamp, multi_episode, multi_step, glo.data,
                                 glo.date, glo.dict, glo.day_data, glo.scaler, glo.min_scaler, flag_lock, sys_model)
        # thread_list[i].daemon=True
    execute_model('i')
    for episode in range(glo.train_times):
        multi_episode.value = episode
        # init model and env
        print("初始化环境")
        execute_model('v')
        for t in range(glo.train_step):
            multi_step.value = t
            # 多进程运行模型
            print("episode:" + str(episode))
            print("     times:" + str(t))
            print("运行模型")
            if execute_model('r'):
                # 当全部pause时
                break
            if sys_model.__contains__("train") or sys_model.__contains__("both"):
                # 执行训练
                print("执行训练")
                if execute_model('t'):
                    break
        # 保存经验池
        if episode % (glo.train_times / glo.save_exp_frequency) == 0 and episode != 0 and sys_model != "run":
            save_experience_pool(ep_dict)
        # 每10个episode保存权重
        # if episode != 0 and episode % 5 == 0:
        execute_model('s')
    # 终止所有进程
    # print("终止进程")
    # execute_model('e')


def run_model():
    global model
    global thread_list
    global thread_flag
    global sys_model
    global time_stamp
    global multi_step
    global multi_episode
    global flag_lock
    # 建立经验池
    ep_dict = {'low': Experience_pool('low'), 'mid': Experience_pool('mid'), 'high': Experience_pool('high')}
    # 载入经验池
    for ep in list(ep_dict.values()):
        ep.load()
    print("建立模型")
    # 建立模型
    for i in range(glo.agent_num):
        thread_list[i] = ACModel(i, model, thread_flag, ep_dict, time_stamp, multi_episode, multi_step, glo.data,
                                 glo.date,
                                 glo.dict, glo.day_data, glo.scaler, glo.min_scaler, flag_lock, sys_model)
        # thread_list[i].daemon=True
    execute_model('i')
    # 当没有全部暂停时
    while not execute_model('r'):
        multi_step.value += 1
        # 循环运行
        pass
    # 否则退出


def save_experience_pool(ep_dict):
    print("经验存储中......请勿退出!!!")
    for ep in list(ep_dict.values()):
        with open("Data/experience_pool_" + str(ep.level) + ".json", "w", encoding='UTF-8') as f:
            exp_list = []
            for i in range(len(ep.exp_pool)):
                exp_list.append(ep.exp_pool[i])
            json.dump(exp_list, f, default=lambda obj: obj.__dict__)
    print("经验存储完成")


if __name__ == '__main__':
    # 全局变量
    model = multiprocessing.Value('u', 'i', lock=False)
    multi_episode = multiprocessing.Value('L', 0)
    multi_step = multiprocessing.Value('L', 0)
    flag_lock = multiprocessing.Lock()
    thread_flag = multiprocessing.Manager().list(range(glo.agent_num))
    thread_list = [None for i in range(glo.agent_num)]
    time_stamp = multiprocessing.Value('L', 0)
    multiprocessing.freeze_support()
    sys_model = input("请输入运行模式：run\\train\\both\n")
    # sys_model = 'both'
    glo.init()
    if sys_model == "both" or sys_model == 'train':
        train_model()
    else:
        run_model()
