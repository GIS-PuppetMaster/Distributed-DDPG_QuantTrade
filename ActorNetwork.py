import glo
from StockSimEnv import Env
from copy import *


class ActorNetwork(object):
    def __init__(self, sess, model, target_model):
        import keras.backend as K
        import tensorflow as tf

        K.set_session(sess)
        self.sess = sess
        self.stock_state_size = glo.stock_state_size
        self.agent_state_size = glo.agent_state_size
        self.action_size = glo.action_size
        self.TAU = glo.tau
        self.LEARNING_RATE = glo.actor_learning_rate
        self.model = model
        self.weights = model.trainable_weights
        self.stock_state = model.input[0]
        self.agent_state = model.input[1]
        self.target_model = target_model
        self.target_weights = target_model.trainable_weights
        self.target_stock_state = target_model.input[0]
        self.target_agent_state = target_model.input[1]
        self.action_gradients = tf.placeholder(tf.float32, [None, glo.action_size])
        params_gradients = tf.gradients(self.model.output, self.weights, -self.action_gradients)
        grad = zip(params_gradients, self.weights)
        global_step = tf.Variable(0, trainable=False)
        learn_rate = tf.train.exponential_decay(self.LEARNING_RATE, global_step, 1000, 0.9)
        self.optimize = tf.train.AdamOptimizer(learn_rate).apply_gradients(grad, global_step=global_step)
        self.sess.run(tf.initialize_all_variables())

    def train(self, stock_state, agent_state, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.stock_state: stock_state,
            self.agent_state: agent_state,
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
        env = Env()
        current_stock_state, current_agent_state = env.get_state()
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
