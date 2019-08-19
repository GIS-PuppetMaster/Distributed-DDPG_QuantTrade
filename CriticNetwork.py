import glo


class CriticNetwork(object):
    def __init__(self, sess, model, target_model):
        import keras.backend as K
        import tensorflow as tf
        K.set_session(sess)
        self.sess = sess
        self.stock_state_size = glo.stock_state_size
        self.agent_state_size = glo.agent_state_size
        self.action_size = glo.action_size
        self.TAU = glo.tau
        self.LEARNING_RATE = glo.critic_learning_rate
        self.model = model
        self.weights = model.trainable_weights
        self.stock_state = model.input[0]
        self.agent_state = model.input[1]
        self.action = model.input[2]
        self.target_model = target_model
        self.target_weights = target_model.trainable_weights
        self.target_stock_state = target_model.input[0]
        self.target_agent_state = target_model.input[1]
        self.target_action = target_model.input[2]
        self.action_grads = tf.gradients(self.model.output, self.action)
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, stock_state, agent_state, action):
        return self.sess.run(self.action_grads, feed_dict={
            self.stock_state: stock_state,
            self.agent_state: agent_state,
            self.action: action
        })[0]

    def update_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.TAU * weights[i] + (1 - self.TAU) * target_weights[i]
        self.target_model.set_weights(target_weights)
