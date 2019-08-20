from multiprocessing import Process
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from Experience import Experience
from StockSimEnv import Env
import tensorflow as tf
import os
import Experience_pool as ep
import numpy as np
import glo
import plotly as py
import plotly.graph_objs as go


class ACModel(Process):
    def __init__(self, index, mode, thread_flag):
        self.index = index
        self.env = Env()
        self.sess = None
        self.actor = None
        self.critic = None
        self.thread_flag = thread_flag
        stock_state, agent_state = self.env.get_state()
        self.stock_state = stock_state
        self.agent_state = agent_state
        self.mode = mode
        self.episode = 0
        Process.__init__(self)

    def init_env(self):
        self.env = Env()

    def init_nn(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽通知信息和警告信息
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        self.sess = sess
        index = self.index
        self.actor = ActorNetwork(sess)
        self.critic = CriticNetwork(sess)
        path = "训练历史权重/Agent编号" + str(index) + "/main_actor_weights.h5"
        if os.path.exists(path):
            self.actor.model.load_weights(path)
        path = "训练历史权重/Agent编号" + str(index) + "/target_actor_weights.h5"
        if os.path.exists(path):
            self.actor.target_model.load_weights(path)
        path = "训练历史权重/Agent编号" + str(index) + "/main_critic_weights.h5"
        if os.path.exists(path):
            self.critic.model.load_weights(path)
        path = "训练历史权重/Agent编号" + str(index) + "/target_critic_weights.h5"
        if os.path.exists(path):
            self.critic.target_model.load_weights(path)

    def train_nn(self):
        stock_state_, agent_state_, action_, reward_, next_stock_state_, next_agent_state_ = ep.get_info_from_experience_list(
            ep.get_experience_batch())
        yi = (np.array(reward_) + glo.gamma * np.array(
            self.critic.target_model.predict([next_stock_state_, next_agent_state_, self.actor.target_model.predict(
                [next_stock_state_, next_agent_state_])]))).tolist()
        # 开始训练
        step_loss = self.critic.model.train_on_batch([stock_state_, agent_state_, action_], [yi])
        a_for_grad = self.actor.model.predict([stock_state_, agent_state_])
        grads = self.critic.gradients(stock_state_, agent_state_, a_for_grad)
        self.actor.train(stock_state_, agent_state_, grads)
        # 更新参数
        self.actor.update_target()
        self.critic.update_target()

    def run_nn(self):
        # 预测前先向网络添加噪声
        self.actor.apply_noise()
        action = self.actor.model.predict([self.stock_state, self.agent_state])[0]
        next_stock_state, next_agent_state, reward = self.env.trade(action)
        if reward is not None:
            experience = Experience(self.stock_state, self.agent_state, action, reward, next_stock_state,
                                    next_agent_state)
            ep.append_experience(experience)

            self.stock_state = next_stock_state
            self.agent_state = next_stock_state
            # 绘图
            if self.index % glo.train_step == 50:
                self.draw_sim_plot(self.env, self.index, self.episode)
        else:
            # 如果下一天是数据边界，则重置环境并退出线程
            self.env.__init__()

    def run(self):
        self.init_nn()
        self.thread_flag[self.index] = 'i'
        print("编号"+str(self.index)+"完成模型初始化")
        while self.mode.value != 'e':
            while self.mode.value != 'r':
                pass
            self.run_nn()
            self.thread_flag[self.index] = 'r'
            print("编号" + str(self.index) + "完成运行")
            while self.mode.value != 't':
                pass
            self.train_nn()
            self.thread_flag[self.index] = 't'
            print("编号" + str(self.index) + "完成训练")

    def draw_sim_plot(self, env, i, episode):
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

    def save_weights(self):
        dis = "训练历史权重/Agent编号" + str(self.index)
        # 目录不存在则创建目录
        if not os.path.exists(dis):
            os.makedirs(dis)

        # 保存最新权重
        self.actor.model.save_weights(dis + '/main_actor_weights.h5', overwrite=True)
        self.actor.target_model.save_weights(dis + '/target_actor_weights.h5', overwrite=True)
        self.critic.model.save_weights(dis + '/main_critic_weights.h5', overwrite=True)
        self.critic.target_model.save_weights(dis + '/target_critic_weights.h5', overwrite=True)
        # 保存历史权重
        self.actor.model.save_weights(dis + '/main_actor_weights_' + str(self.episode) + '.h5', overwrite=True)
        self.actor.target_model.save_weights(dis + '/target_actor_weights_' + str(self.episode) + '.h5', overwrite=True)
        self.critic.model.save_weights(dis + '/main_critic_weights_' + str(self.episode) + '.h5', overwrite=True)
        self.critic.target_model.save_weights(dis + '/target_critic_weights_' + str(self.episode) + '.h5',
                                              overwrite=True)
