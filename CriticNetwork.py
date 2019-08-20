import glo


class CriticNetwork(object):
    def __init__(self, sess):
        import keras.backend as K
        import tensorflow as tf
        K.set_session(sess)
        self.sess = sess
        self.stock_state_size = glo.stock_state_size
        self.agent_state_size = glo.agent_state_size
        self.action_size = glo.action_size
        self.TAU = glo.tau
        self.LEARNING_RATE = glo.critic_learning_rate
        self.model, self.weights, self.stock_state, self.agent_state, self.action = self.build_critic_network()
        self.target_model, self.target_weights, self.target_stock_state, self.target_agent_state, self.target_action = self.build_critic_network()
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

    def build_critic_network(self):
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
        return model, model.trainable_weights, input_stock_state, input_agent_state, input_action
