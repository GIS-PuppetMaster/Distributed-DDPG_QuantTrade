import glo
import os
from StockSimEnv import Env
import random
import math
import json
from stock_date import *
import plotly as py
import plotly.graph_objs as go
import sys
from Experience import Experience
from ACModel import ACModel
import multiprocessing
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork

# f = open('trade.log', 'a')
# sys.stdout = f
# sys.stderr = f
# tf预设置


def append_experience(experience, experience_p, cursor):
    """
    向经验池中添加经验，如果经验池已满，则从index=0开始覆盖
    :param experience: 要添加的经验
    :return: None
    """
    c = int(cursor["cursor"])
    c = (c + 1) % glo.experience_pool_size
    experience_p[c] = experience
    cursor["cursor"] = c


def get_experience_batch(experience_p):
    """
    获取经验训练包
    :return: 从经验池中随机采样的经验训练包
    """
    return random.sample(experience_p, glo.mini_batch_size)


def execute_action(env, action):
    """
    执行action
    :param env: 执行动作的环境
    :param action: 动作
    :return: 执行动作后的状态和奖励和实际交易量
    """
    return env.trade(action)


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


def build_actor_network():
    from keras.models import Model
    from keras.layers import Input, SeparableConv2D, Activation, BatchNormalization, Dense, Concatenate, Flatten
    from keras.utils import plot_model
    from keras.optimizers import Adam
    """
       输入：state(stock,agent)
       输出：action
       loss：max(q)，即-tf.reduce_mean(q)
       :return:actor_net_model,weights,stock_state,agent_state
       """
    input_stock_state = Input(shape=(glo.day, glo.stock_state_size, glo.count))
    # input_stock_state_ = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input_stock_state)
    input_agent_state = Input(shape=(glo.agent_state_size,))
    # input_agent_state_ = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input_agent_state)
    x_stock_state = SeparableConv2D(filters=1, kernel_size=6, padding='valid', data_format='channels_first')(
        input_stock_state)
    x_stock_state = Activation('tanh')(x_stock_state)
    x_stock_state = BatchNormalization(axis=2, epsilon=1e-4, scale=True, center=True)(x_stock_state)
    x_stock_state = Flatten()(x_stock_state)
    dense01 = Dense(16)(x_stock_state)
    dense01 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense01)
    dense01 = Activation('tanh')(dense01)
    dense01 = Dense(8)(dense01)
    dense01 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense01)
    dense01 = Activation('tanh')(dense01)
    merge_layer = Concatenate()([dense01, input_agent_state])
    dense02 = Dense(32)(merge_layer)
    dense02 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense02)
    dense02 = Activation('tanh')(dense02)
    dense02 = Dense(32)(dense02)
    dense02 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense02)
    dense02 = Activation('tanh')(dense02)
    dense02 = Dense(8)(dense02)
    dense02 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense02)
    dense02 = Activation('tanh')(dense02)
    output = Dense(glo.action_size, name='output', activation='tanh')(dense02)
    model = Model(inputs=[input_stock_state, input_agent_state], outputs=[output])
    plot_model(model, to_file='actor_net.png', show_shapes=True)
    return model


def build_critic_network():
    from keras.models import Model
    from keras.layers import Input, SeparableConv2D, Activation, BatchNormalization, Dense, Concatenate, Flatten
    from keras.utils import plot_model
    from keras.optimizers import Adam
    """
       输入：state,action
       输出：q
       loss：(reward+gamma*q_)-q
       :return:
       """
    input_stock_state = Input(shape=(glo.day, glo.stock_state_size, glo.count))
    # input_stock_state_ = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input_stock_state)
    input_agent_state = Input(shape=(glo.agent_state_size,))
    # input_agent_state_ = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input_agent_state)
    input_action = Input(shape=(glo.action_size,))
    # x_stock_state = Conv1D(filters=25, kernel_size=2, padding='same')(input_stock_state_)
    # x_stock_state = BatchNormalization(axis=2,epsilon=1e-4, scale=True, center=True)(x_stock_state)
    x_stock_state = Flatten()(input_stock_state)
    # x_stock_state = Activation('tanh')(x_stock_state)
    dense01 = Dense(64)(x_stock_state)
    # dense01 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense01)
    dense01 = Activation('tanh')(dense01)
    dense01 = Dense(8)(dense01)
    # dense01 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense01)
    dense01 = Activation('tanh')(dense01)
    merge_layer = Concatenate()([dense01, input_agent_state, input_action])
    dense02 = Dense(8)(merge_layer)
    # dense02 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense02)
    dense02 = Activation('tanh')(dense02)
    dense02 = Dense(4)(dense02)
    # dense02 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense02)
    dense02 = Activation('tanh')(dense02)
    # q
    output = Dense(glo.action_size, name='output', activation='tanh')(dense02)
    model = Model(inputs=[input_stock_state, input_agent_state, input_action], outputs=[output])
    model.compile(optimizer=Adam(glo.critic_learning_rate), loss='mse')
    plot_model(model, to_file='critic_net.png', show_shapes=True)
    return model


def run_nn(acmodel, i, episode, experience_pool, cursor, lock):
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    actor = ActorNetwork(sess, acmodel.actor, acmodel.target_actor)
    env = acmodel.env
    current_stock_state = acmodel.stock_state
    current_agent_state = acmodel.agent_state
    # 预测前先向网络添加噪声
    actor.apply_noise()
    action = actor.model.predict([current_stock_state, current_agent_state])[0]
    next_stock_state, next_agent_state, reward = execute_action(env, action)
    if reward is not None:
        experience = Experience(current_stock_state, current_agent_state, action, reward, next_stock_state,
                                next_agent_state)
        lock.acquire()
        append_experience(experience, experience_pool, cursor)
        lock.release()
        acmodel.stock_state = next_stock_state
        acmodel.agent_state = next_stock_state
        # 绘图
        if i % glo.train_step == 50:
            draw_sim_plot(env, i, episode)
    else:
        # 如果下一天是数据边界，则重置环境并退出线程
        env.__init__()


def train_nn(acmodel, experience_p, i, agent_list):
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    actor = ActorNetwork(sess, acmodel.actor, acmodel.target_actor)
    critic = CriticNetwork(sess, acmodel.critic, acmodel.target_critic)
    stock_state_, agent_state_, action_, reward_, next_stock_state_, next_agent_state_ = get_info_from_experience_list(
        get_experience_batch(experience_p))
    yi = (np.array(reward_) + glo.gamma * np.array(
        critic.target_model.predict([next_stock_state_, next_agent_state_, actor.target_model.predict(
            [next_stock_state_, next_agent_state_])]))).tolist()
    # 开始训练
    step_loss = critic.model.train_on_batch([stock_state_, agent_state_, action_], [yi])
    a_for_grad = actor.model.predict([stock_state_, agent_state_])
    grads = critic.gradients(stock_state_, agent_state_, a_for_grad)
    actor.train(stock_state_, agent_state_, grads)
    # 更新参数
    actor.update_target()
    critic.update_target()
    agent_list[i].actor = actor.model
    agent_list[i].target_actor = actor.target_model
    agent_list[i].critic = critic.model
    agent_list[i].target_critic = critic.target_model


def init_nn(i, agent_list, lock):
    import tensorflow as tf
    with tf.variable_scope(str(i)):
        print("初始化" + str(i) + "号神经网")
        actor = build_actor_network()
        target_actor = build_actor_network()
        critic = build_critic_network()
        target_critic = build_critic_network()
        # 如果权重存在则载入权重
        path = "训练历史权重/Agent编号" + str(i) + "/main_actor_weights.h5"
        if os.path.exists(path):
            actor.load_weights(path)
        path = "训练历史权重/Agent编号" + str(i) + "/target_actor_weights.h5"
        if os.path.exists(path):
            target_actor.load_weights(path)
        path = "训练历史权重/Agent编号" + str(i) + "/main_critic_weights.h5"
        if os.path.exists(path):
            critic.load_weights(path)
        path = "训练历史权重/Agent编号" + str(i) + "/target_critic_weights.h5"
        if os.path.exists(path):
            target_critic.load_weights(path)
        env = Env()
        current_stock_state, current_agent_state = env.get_state()
        print("添加到列表"+str(i))
        lock.acquire()
        agent_list.append(
            ACModel(actor, target_actor, critic, target_critic, env, current_stock_state, current_agent_state))
        lock.release()
        print("添加完毕"+str(i))


def run_model():
    global experience_pool
    global agent_list
    global sys_model

    # 载入经验池
    if os.path.exists("Data/experience_pool.json"):
        with open("Data/experience_pool.json", "r", encoding="UTF-8") as f:
            s = f.read()
            experience_pool = json.loads(s, object_hook=Experience.object_hook)
        print("已载入经验池")
    lock = multiprocessing.Lock()
    for episode in range(glo.train_times):
        # init model and env

        jobs = [multiprocessing.Process(target=init_nn, args=(i, agent_list, lock)) for i in
                range(glo.agent_num)]
        for k in jobs:
            k.start()
        for k in jobs:
            k.join()
        print("agent_list_len:" + str(len(agent_list)))
        for t in range(glo.train_step):
            # 多线程运行模型
            flag = False
            while len(experience_pool) <= glo.experience_pool_size:
                if len(experience_pool) < glo.experience_pool_size:
                    # 观察环境模式
                    flag = True
                    print("经验池大小：" + str(len(experience_pool)))
                jobs = [multiprocessing.Process(target=run_nn, args=(
                    agent_list[i], i, episode, experience_pool, experience_cursor, lock)) for i in range(glo.agent_num)]
                for k in jobs:
                    k.start()
                for k in jobs:
                    k.join()

                if not flag:
                    break
            # 如果刚刚从观察模式跳出
            if flag:
                # 保存经验池
                save_experience_pool()
            if sys_model.__contains__("train") or sys_model.__contains__("both"):
                # 执行训练
                jobs = [multiprocessing.Process(target=train_nn, args=(agent_list[i], experience_pool, i, agent_list))
                        for i in
                        range(glo.agent_num)]
                for k in jobs:
                    k.start()
                for k in jobs:
                    k.join()
        # 保存经验池
        if episode % glo.train_times == int(glo.train_times / 10):
            save_experience_pool()
            # 保存权重
            for i in range(glo.agent_num):
                dis = "训练历史权重/Agent编号" + str(i)
                # 目录不存在则创建目录
                if not os.path.exists(dis):
                    os.makedirs(dis)
                main_actor_net = agent_list[i].actor
                target_actor_net = agent_list[i].target_actor
                main_critic_net = agent_list[i].critic
                target_critic_net = agent_list[i].target_critic

                # 保存最新权重
                main_actor_net.save_weights(dis + '/main_actor_weights.h5', overwrite=True)
                target_actor_net.save_weights(dis + '/target_actor_weights.h5', overwrite=True)
                main_critic_net.save_weights(dis + '/main_critic_weights.h5', overwrite=True)
                target_critic_net.save_weights(dis + '/target_critic_weights.h5', overwrite=True)
                # 保存历史权重
                main_actor_net.save_weights(dis + '/main_actor_weights_' + str(episode) + '.h5', overwrite=True)
                target_actor_net.save_weights(dis + '/target_actor_weights_' + str(episode) + '.h5', overwrite=True)
                main_critic_net.save_weights(dis + '/main_critic_weights_' + str(episode) + '.h5', overwrite=True)
                target_critic_net.save_weights(dis + '/target_critic_weights_' + str(episode) + '.h5', overwrite=True)


def draw_sim_plot(env, i, episode):
    time_list = env.time_list
    profit_list = env.profit_list
    reference_list = env.reference_list
    price_list = env.price_list
    quant_list = np.array(env.stock_value)[:, 1]
    amount_list = []
    amount = env.stock_value[0][1]
    for l in env.stock_value:
        amount += l[1]
        amount_list.append(amount)
    dis = "运行结果/Agent编号" + str(i)
    path = dis + "/episode_" + str(episode) + ".html"
    # 目录不存在则创建目录
    if not os.path.exists(dis):
        os.makedirs(dis)
    profit_scatter = go.Scatter(x=time_list,
                                y=profit_list,
                                name='MADDPG',
                                line=dict(color='red'),
                                mode='lines')
    reference_scatter = go.Scatter(x=time_list,
                                   y=reference_list,
                                   name='基准',
                                   line=dict(color='blue'),
                                   mode='lines')
    price_scatter = go.Scatter(x=time_list,
                               y=price_list,
                               name='股价',
                               line=dict(color='orange'),
                               mode='lines',
                               xaxis='x',
                               yaxis='y2',
                               opacity=1)
    trade_bar = go.Bar(x=time_list,
                       y=quant_list,
                       name='交易量（手）',
                       marker_color='#000099',
                       xaxis='x',
                       yaxis='y3',
                       opacity=0.3)
    amount_scatter = go.Scatter(x=time_list,
                                y=amount_list,
                                name='持股数量（手）',
                                line=dict(color='rgba(0,204,255,0.6)'),
                                mode='lines',
                                fill='tozeroy',
                                fillcolor='rgba(0,204,255,0.3)',
                                xaxis='x',
                                yaxis='y4',
                                opacity=0.6)
    """
    cl = np.array(candle_list)
    price_candle = go.Candlestick(x=time_list,
                                  xaxis='x',
                                  yaxis='y2',
                                  name='价格',
                                  open=cl[:, 0],
                                  close=cl[:, 1],
                                  high=cl[:, 2],
                                  low=cl[:, 3],
                                  increasing=dict(line=dict(color='#FF2131')),
                                  decreasing=dict(line=dict(color='#00CCFF')))
    """
    py.offline.plot({
        "data": [profit_scatter, reference_scatter, price_scatter, trade_bar,
                 amount_scatter],
        "layout": go.Layout(
            title=env.stock_code + "回测结果" + "     初始资金：" + str(env.ori_money) + "     初始股票价值" + str(
                env.ori_value) + "    总初始本金:" + str(env.ori_money + env.ori_value),
            xaxis=dict(title='日期', type="category", showgrid=False, zeroline=False),
            yaxis=dict(title='收益率', showgrid=False, zeroline=False),
            yaxis2=dict(title='股价', overlaying='y', side='right',
                        titlefont={'color': 'orange'}, tickfont={'color': 'orange'},
                        showgrid=False,
                        zeroline=False),
            yaxis3=dict(title='交易量', overlaying='y', side='right',
                        titlefont={'color': '#000099'}, tickfont={'color': '#000099'},
                        showgrid=False, position=0.97, zeroline=False, anchor='free'),
            yaxis4=dict(title='持股量', overlaying='y', side='left',
                        titlefont={'color': '#00ccff'}, tickfont={'color': '#00ccff'},
                        showgrid=False, position=0.03, zeroline=False, anchor='free'),
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
        )
    }, auto_open=False, filename=path)


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
        json.dump(experience_pool, f, default=lambda obj: obj.__dict__)
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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽通知信息和警告信息
    experience_pool = multiprocessing.Manager().list(range(0))
    agent_list = multiprocessing.Manager().list(range(0))
    experience_cursor = multiprocessing.Manager().dict()
    experience_cursor["cursor"] = 0
    # multiprocessing.freeze_support()
    sys_model = input("请输入运行模式：run\\train\\both\n")
    run_model()
