from multiprocessing import Process
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from Experience import Experience
from StockSimEnv import Env
import os
import numpy as np
import glo
import plotly as py
import plotly.graph_objs as go
import random
import sys
from datetime import *


class ACModel(Process):
    def __init__(self, index, mode, thread_flag, ep, time_stamp, episode, step, _data, _date, _dict, _scaler, lock,
                 obs):
        self.index = index
        self.env = Env(stock_code='000517.XSHE', scaler=_scaler)
        self.sess = None
        self.actor = None
        self.critic = None
        self.thread_flag = thread_flag
        self.ep = ep
        self.stock_state, self.agent_state = self.env.get_state()
        self.mode = mode
        self.episode = episode
        self.step = step
        self.time_stamp = time_stamp
        self.last_stamp = -1
        self.data = _data
        self.date = _date
        self.dict = _dict
        self.lock = lock
        self.obs = obs
        self.reset_counter = 0
        self.step_loss_list = []
        self.reward_list = []
        self.pause = False
        self.scaler = _scaler
        Process.__init__(self)

    def init_env(self):
        self.env = Env(stock_code='000517.XSHE', scaler=self.scaler)

    def init_nn(self):
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽通知信息和警告信息
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        self.sess = sess
        index = self.index
        self.actor = ActorNetwork(sess)
        self.critic = CriticNetwork(sess)
        path = "最新训练权重/Agent编号" + str(index) + "/main_actor_weights.h5"
        if os.path.exists(path):
            print("载入权重main_actor_weights.h5")
            self.actor.model.load_weights(path)
        path = "最新训练权重/Agent编号" + str(index) + "/target_actor_weights.h5"
        if os.path.exists(path):
            print("载入权重target_actor_weights.h5")
            self.actor.target_model.load_weights(path)
        path = "最新训练权重/Agent编号" + str(index) + "/main_critic_weights.h5"
        if os.path.exists(path):
            print("载入权重main_critic_weights.h5")
            self.critic.model.load_weights(path)
        path = "最新训练权重/Agent编号" + str(index) + "/target_critic_weights.h5"
        if os.path.exists(path):
            print("载入权重target_critic_weights.h5")
            self.critic.target_model.load_weights(path)
        """
        os.remove('日志/')
        os.makedirs('日志/')
        f = open('日志/'+str(self.index)+'_trade.log', 'a')
        sys.stdout = f
        sys.stderr = f
        """

    def train_nn(self):
        # print("编号" + str(self.index) + "获取经验包")
        stock_state_, agent_state_, action_, reward_, next_stock_state_, next_agent_state_ = self.ep.get_info_from_experience_list(
            self.ep.get_experience_batch())
        # print("编号" + str(self.index) + "生成yi")
        yi = (np.array(reward_) + glo.gamma * np.array(
            self.critic.target_model.predict([next_stock_state_, next_agent_state_, self.actor.target_model.predict(
                [next_stock_state_, next_agent_state_])]))).tolist()
        # 开始训练
        # print("编号" + str(self.index) + "开始训练critic")
        step_loss = self.critic.model.train_on_batch([stock_state_, agent_state_, action_], [yi])
        self.step_loss_list.append(step_loss)
        # print("编号" + str(self.index) + "生成a_for_grad")
        a_for_grad = self.actor.model.predict([stock_state_, agent_state_])
        # print("编号" + str(self.index) + "生成梯度")
        grads = self.critic.gradients(stock_state_, agent_state_, a_for_grad)
        # print("编号" + str(self.index) + "梯度\n" + str(grads))
        # print("编号" + str(self.index) + "训练actor")
        self.actor.train(stock_state_, agent_state_, grads)
        # print("编号" + str(self.index) + "更新target")
        # 更新参数
        self.actor.update_target()
        self.critic.update_target()

    def run_nn(self):
        # 预测前先向网络添加噪声
        # print("编号" + str(self.index) + "添加噪声")
        if self.obs.value == 'f':
            self.actor.apply_noise()
            # print("编号" + str(self.index) + "预测动作")
            action = self.actor.model.predict([self.stock_state, self.agent_state])[0]
            print("编号" + str(self.index) + "action:" + str(action))
            # TODO:完成ES噪声算法后删除
            # print("编号" + str(self.index) + "添加噪声")
            action += np.random.randn(2, ) * 0.01
            if action[0] > 1:
                action[0] = 1
            if action[0] < -1:
                action[0] = -1
            if action[1] > 1:
                action[1] = 1
            if action[1] < -1:
                action[1] = -1
            # print("编号" + str(self.index) + "action_noise:" + str(action))
        else:
            action = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        # print("编号" + str(self.index) + "进行交易")
        next_stock_state, next_agent_state, reward = self.env.trade(action)
        # print("编号" + str(self.index) + "reward:" + str(reward))
        if reward is not None:
            # 解决numpy.float32没有__dict__方法，使得经验无法使用Json.dump的问题
            # print("编号" + str(self.index) + "append经验")
            a = []
            for i in range(glo.action_size):
                a.append(action[i].item())
            experience = Experience(self.stock_state, self.agent_state, [a], [[float(str(reward))]], next_stock_state,
                                    next_agent_state)
            self.ep.append_experience(experience)
            self.reward_list.append(reward)
            self.stock_state = next_stock_state
            self.agent_state = next_agent_state
            # 绘图
            if self.step.value == glo.train_step - 1:
                # print("编号" + str(self.index) + "绘图")
                self.draw_sim_plot(self.env, self.index, self.episode.value)
        else:
            # self.env.reset()
            # print("编号:" + str(self.index) + " 进入pause模式")
            self.draw_sim_plot(self.env, self.index, self.episode.value)
            self.pause = True
            # self.stock_state, self.agent_state = self.env.get_state()
        # 连续15轮训练交易次数小于20次则重置网络
        if self.step.value == glo.train_step - 1 or (self.env.start_date - self.env.gdate.get_date()).days == 365:
            if len(self.env.stock_value) <= glo.reset_trigger:
                self.reset_counter += 1
            else:
                self.reset_counter = 0
        if self.reset_counter == glo.reset:
            print("编号" + str(self.index) + "重置网络")
            self.actor = ActorNetwork(self.sess)
            self.critic = CriticNetwork(self.sess)
            self.reset_counter = 0
            dir = 'log/Agent编号' + str(self.index)
            if not os.path.exists(dir):
                os.makedirs(dir)
            with open(dir + "/episode" + str(self.episode.value), "a") as f:
                f.write("step:" + str(self.step.value) + "重置网络\n")
        if (self.env.start_date - self.env.gdate.get_date()).days >= 365:
            self.pause = True

    def run(self):
        glo.data = self.data
        glo.date = self.date
        glo.dict = self.dict
        self.data = None
        self.date = None
        self.dict = None
        while self.mode.value != 'e':
            """
            print("mode:"+str(self.mode.value))
            print("last_stamp:"+str(self.last_stamp))
            print("time_stamp:"+str(self.time_stamp.value))
            """
            if self.mode.value == 'i' and self.time_stamp.value != self.last_stamp:
                self.init_nn()
                self.lock.acquire()
                self.last_stamp = int(self.time_stamp.value)
                self.thread_flag[self.index] = 'i'
                self.lock.release()
                # print("编号" + str(self.index) + "完成模型初始化")
            elif self.mode.value == 'r' and self.time_stamp.value != self.last_stamp:
                if not self.pause:
                    self.run_nn()
                else:
                    pass
                    print("编号:" + str(self.index) + " pause")
                self.lock.acquire()
                self.last_stamp = int(self.time_stamp.value)
                if not self.pause:
                    self.thread_flag[self.index] = 'r'
                else:
                    self.thread_flag[self.index] = 'p'
                self.lock.release()
                # print("编号" + str(self.index) + "完成运行")
            elif self.mode.value == 't' and self.time_stamp.value != self.last_stamp:
                if not self.pause:
                    self.train_nn()
                else:
                    pass
                    print("编号:" + str(self.index) + " pause")
                self.lock.acquire()
                self.last_stamp = int(self.time_stamp.value)
                if not self.pause:
                    self.thread_flag[self.index] = 't'
                else:
                    self.thread_flag[self.index] = 'p'
                self.lock.release()
                # print("编号" + str(self.index) + "完成训练")
            elif self.mode.value == 'v' and self.time_stamp.value != self.last_stamp:
                self.init_env()
                self.lock.acquire()
                self.last_stamp = int(self.time_stamp.value)
                self.thread_flag[self.index] = 'v'
                self.lock.release()
                # 重置环境后取消pause模式
                self.pause = False
            elif self.mode.value == 's' and self.time_stamp.value != self.last_stamp:
                self.save_weights()
                self.lock.acquire()
                self.last_stamp = int(self.time_stamp.value)
                self.thread_flag[self.index] = 's'
                self.lock.release()

    def draw_sim_plot(self, env, index, episode):
        time_list = env.time_list
        profit_list = env.profit_list
        reference_list = env.reference_list
        price_list = env.price_list
        quant_list = np.array(env.stock_value)[1:, 1]
        amount_list = []
        amount = env.stock_value[0][1]
        for i in range(len(env.stock_value)):
            if i == 0:
                continue
            else:
                l = env.stock_value[i]
                amount += l[1]
                amount_list.append(amount)
        dis = "E:/运行结果/Agent编号" + str(index)
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
                title=env.stock_code + " 编号" + str(self.index) + " 回测结果" + "     初始资金：" + str(
                    env.ori_money) + "     初始股票价值" + str(
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
                plot_bgcolor='#FFFFFF'
            )
        }, auto_open=False, filename=path)
        loss_scatter = go.Scatter(x=[i for i in range(len(self.step_loss_list))],
                                  y=self.step_loss_list,
                                  name='loss',
                                  line=dict(color='orange'),
                                  mode='lines',
                                  opacity=1)
        path = dis + "/loss.html"
        py.offline.plot({
            "data": [loss_scatter],
            "layout": go.Layout(
                title="loss 编号" + str(self.index),
                xaxis=dict(title='训练次数', showgrid=False, zeroline=False),
                yaxis=dict(title='loss', showgrid=False, zeroline=False),
                paper_bgcolor='#FFFFFF',
                plot_bgcolor='#FFFFFF'
            )
        }, auto_open=False, filename=path)
        reward_scatter = go.Scatter(x=[i for i in range(len(self.reward_list))],
                                  y=self.reward_list,
                                  name='reward',
                                  line=dict(color='orange'),
                                  mode='lines',
                                  opacity=1)
        path = dis + "/reward.html"
        py.offline.plot({
            "data": [reward_scatter],
            "layout": go.Layout(
                title="reward 编号" + str(self.index),
                xaxis=dict(title='训练次数', showgrid=False, zeroline=False),
                yaxis=dict(title='reward', showgrid=False, zeroline=False),
                paper_bgcolor='#FFFFFF',
                plot_bgcolor='#FFFFFF'
            )
        }, auto_open=False, filename=path)

    def save_weights(self):
        dis = "最新训练权重/Agent编号" + str(self.index)
        # 目录不存在则创建目录
        if not os.path.exists(dis):
            os.makedirs(dis)

        # 保存最新权重
        self.actor.model.save_weights(dis + '/main_actor_weights.h5', overwrite=True)
        self.actor.target_model.save_weights(dis + '/target_actor_weights.h5', overwrite=True)
        self.critic.model.save_weights(dis + '/main_critic_weights.h5', overwrite=True)
        self.critic.target_model.save_weights(dis + '/target_critic_weights.h5', overwrite=True)
        # 保存历史权重
        dis = "E:/训练历史权重/Agent编号" + str(self.index)
        # 目录不存在则创建目录
        if not os.path.exists(dis):
            os.makedirs(dis)

        self.actor.model.save_weights(dis + '/' + str(self.episode.value) + '_main_actor_weights.h5', overwrite=True)
        self.actor.target_model.save_weights(dis + '/' + str(self.episode.value) + '_target_actor_weights.h5',
                                             overwrite=True)
        self.critic.model.save_weights(dis + '/' + str(self.episode.value) + '_main_critic_weights.h5', overwrite=True)
        self.critic.target_model.save_weights(dis + '/' + str(self.episode.value) + '_target_critic_weights.h5',
                                              overwrite=True)
