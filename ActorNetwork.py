import glo
from StockSimEnv import Env
from copy import *


class ActorNetwork(object):
    def __init__(self, sess):
        import keras.backend as K
        import tensorflow as tf

        K.set_session(sess)
        self.sess = sess
        self.stock_state_size = glo.stock_state_size
        self.agent_state_size = glo.agent_state_size
        self.price_state_size = glo.price_state_size
        self.action_size = glo.action_size
        self.TAU = glo.tau
        self.LEARNING_RATE = glo.actor_learning_rate
        self.model, self.weights, self.stock_state, self.agent_state, self.price_state = self.build_actor_network()
        self.target_model, self.target_weights, self.target_stock_state, self.target_agent_state, self.target_price_state = self.build_actor_network()

        self.action_gradients = tf.placeholder(tf.float32, [None, glo.action_size])
        params_gradients = tf.gradients(self.model.output, self.weights, -self.action_gradients)
        grad = zip(params_gradients, self.weights)
        global_step = tf.Variable(0, trainable=False)
        learn_rate = tf.train.exponential_decay(self.LEARNING_RATE, global_step, 2000, 0.9)
        self.optimize = tf.train.AdamOptimizer(learn_rate).apply_gradients(grad, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())

    def train(self, stock_state, agent_state, price_state, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.stock_state: stock_state,
            self.agent_state: agent_state,
            self.price_state: price_state,
            self.action_gradients: action_grads
        })

    def update_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.TAU * weights[i] + (1 - self.TAU) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def f(self, w):
        temp_actor = deepcopy(self.model)
        temp_actor.set_weights(w)
        # TODO 传入scaler
        # TODO price_state处理
        env = Env()
        current_stock_state, current_agent_state, current_price_state = env.get_state()
        reward = 0
        for i in range(20):
            action = temp_actor.predict([current_stock_state, current_agent_state])[0]
            next_stock_state, next_agent_state, r = env.trade(action)
            reward += r
            current_stock_state = next_stock_state
            current_agent_state = next_agent_state
        return reward

    def apply_noise(self):
        pass
        """
        # 会导致train后权重为Nan
        import numpy as np
        weights = self.model.get_weights()
        for i in range(len(weights)):
            weights[i] += np.random.standard_normal(weights[i].shape)*0.05
            # weights[i] += np.zeros(weights[i].shape)
        self.model.set_weights(weights)
        """
        """
        # 采用ES算法自适应高斯噪声
        weights = self.model.get_weights()
        length = len(weights)
        for i in range(50):
            N = np.random.randn(glo.npop, length)
            R = np.zeros(glo.npop)
            for j in range(glo.npop):
                w_try = weights + glo.sigma * N[j]
                R[j] = self.f(w_try)
            A = (R - np.mean(R)) / np.std(R)
            weights += glo.alpha / (glo.npop * glo.sigma) * np.dot(N.T, A)
        self.model.set_weights(weights)
        """

    def build_actor_network(self):
        from keras.models import Model
        from keras.layers import Input, Conv1D, Activation, BatchNormalization, Dense, Concatenate, Flatten, \
            regularizers, Reshape, LSTM, Bidirectional
        from keras.utils import plot_model
        """
           输入：state(stock,agent)
           输出：action
           loss：max(q)，即-tf.reduce_mean(q)
           :return:actor_net_model,weights,stock_state,agent_state
           """
        input_stock_state = Input(shape=(glo.day, glo.stock_state_size))
        # input_stock_state_ = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input_stock_state)
        input_agent_state = Input(shape=(glo.agent_state_size,))
        input_agent_state_ = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input_agent_state)
        input_price_state = Input(shape=(glo.price_state_size,))
        input_price_state_ = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input_price_state)
        """
        # 首先把日期时序和特征压缩
        x_stock_state = Reshape((glo.count, glo.day * glo.stock_state_size))(input_stock_state)
        # 对分钟时序进行卷积
        x_stock_state = Conv1D(filters=glo.day*glo.stock_state_size, kernel_size=glo.count, padding='valid')(x_stock_state)
        x_stock_state = BatchNormalization(axis=2, epsilon=1e-4, scale=True, center=True)(x_stock_state)
        x_stock_state = Activation('tanh')(x_stock_state)
        # 展开日期时序和特征
        x_stock_state = Reshape((glo.day, glo.stock_state_size))(x_stock_state)
        """
        x_stock_state = Bidirectional(
            LSTM(32, activation='tanh', recurrent_activation='tanh', kernel_regularizer=regularizers.l2(0.01),
                 unroll=True, return_sequences=True))(input_stock_state)
        x_stock_state = Bidirectional(
            LSTM(16, activation='tanh', recurrent_activation='tanh', kernel_regularizer=regularizers.l2(0.01),
                 unroll=True, return_sequences=False))(x_stock_state)
        # x_stock_state = Flatten()(input_stock_state)
        dense01 = Dense(32, kernel_regularizer=regularizers.l2(0.01))(x_stock_state)
        dense01 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense01)
        dense01 = Activation('tanh')(dense01)
        dense01 = Dense(8, kernel_regularizer=regularizers.l2(0.01))(dense01)
        dense01 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense01)
        dense01 = Activation('tanh')(dense01)
        merge_layer = Concatenate()([dense01, input_agent_state_, input_price_state_])
        dense02 = Dense(32)(merge_layer)
        dense02 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense02)
        dense02 = Activation('tanh')(dense02)
        dense02 = Dense(16, kernel_regularizer=regularizers.l2(0.01))(dense02)
        dense02 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense02)
        dense02 = Activation('tanh')(dense02)
        dense02 = Dense(8, kernel_regularizer=regularizers.l2(0.01))(dense02)
        dense02 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense02)
        dense02 = Activation('tanh')(dense02)
        output = Dense(glo.action_size, name='output', activation='tanh')(dense02)
        model = Model(inputs=[input_stock_state, input_agent_state,input_price_state], outputs=[output])
        plot_model(model, to_file='actor_net.png', show_shapes=True)
        return model, model.trainable_weights, input_stock_state, input_agent_state, input_price_state
