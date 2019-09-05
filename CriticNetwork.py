import glo


class CriticNetwork(object):
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
        self.LEARNING_RATE = glo.critic_learning_rate
        self.model, self.weights, self.stock_state, self.agent_state, self.price_state, self.action = self.build_critic_network()
        self.target_model, self.target_weights, self.target_stock_state, self.target_agent_state, self.target_price_state, self.target_action = self.build_critic_network()
        self.action_grads = tf.gradients(self.model.output, self.action)
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, stock_state, agent_state, price_state, action):
        return self.sess.run(self.action_grads, feed_dict={
            self.stock_state: stock_state,
            self.agent_state: agent_state,
            self.price_state: price_state,
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
        from keras.layers import Input, Conv1D, Activation, BatchNormalization, Dense, Concatenate, Flatten, \
            Reshape, LSTM, Bidirectional, GaussianNoise, CuDNNLSTM, SeparableConv1D, AveragePooling1D
        from keras.utils import plot_model
        from keras import regularizers
        from keras.optimizers import Adam
        from MyKerasTool import Dense_res_block3
        from MyKerasTool import Dense_layer_connect
        from MyKerasTool import Conv1D_conv_block
        from MyKerasTool import Conv1D_identity_block
        from MyKerasTool import CuDNNLSTM_res_block2
        from MyKerasTool import CuDNNLSTM_res_block3
        from MyKerasTool import Dense_BN
        """
           输入：state,action
           输出：q
           loss：(reward+gamma*q_)-q
           :return:
           """
        input_stock_state = Input(shape=(glo.day, glo.stock_state_size))
        input_agent_state = Input(shape=(glo.agent_state_size,))
        input_agent_state_ = BatchNormalization(axis=1, epsilon=1e-4, scale=True, center=True)(input_agent_state)
        input_price_state = Input(shape=(glo.price_state_size,))
        input_action = Input(shape=(glo.action_size,))
        x_stock_state = CuDNNLSTM(64, kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(
            input_stock_state)
        x_stock_state = BatchNormalization(epsilon=1e-4, scale=True, center=True)(x_stock_state)
        x_stock_state = Activation('tanh')(x_stock_state)

        x_stock_state = Conv1D_conv_block(x_stock_state, filters=(32, 8, 32), block_name='stage1_conv-')
        for i in range(2):
            x_stock_state = Conv1D_identity_block(x_stock_state, filters=(32, 16, 32), block_name='stage1_identity_' + str(i) + '-')

        x_stock_state = Conv1D_conv_block(x_stock_state, filters=(16, 4, 16), strides=2, block_name='stage2_conv-')
        for i in range(3):
            x_stock_state = Conv1D_identity_block(x_stock_state, filters=(16, 8, 16), block_name='stage2_identity_' + str(i) + '-')

        # x_stock_state = Dense_layer_connect(x_stock_state, units=16)

        x_stock_state = Conv1D_conv_block(x_stock_state, filters=(16, 4, 16), strides=2, block_name='stage3_conv-')
        for i in range(5):
            x_stock_state = Conv1D_identity_block(x_stock_state, filters=(8, 8, 16), block_name='stage3_identity_' + str(i) + '-')

        x_stock_state = Conv1D_conv_block(x_stock_state, filters=(16, 8, 8), strides=2, block_name='stage4_conv-')
        for i in range(2):
            x_stock_state = Conv1D_identity_block(x_stock_state, filters=(4, 4, 8), block_name='stage4_identity_' + str(i) + '-')

        x_stock_state = AveragePooling1D(pool_size=4, strides=2, data_format='channels_first')(x_stock_state)
        x_stock_state = Flatten()(x_stock_state)
        merge = Concatenate()([x_stock_state, input_price_state, input_agent_state_, input_action])
        layer = Dense_layer_connect(merge, units=16)
        layer = Dense_BN(layer, units=32)
        layer = Dense_BN(layer, units=8)
        output = Dense(1)(layer)
        model = Model(inputs=[input_stock_state, input_agent_state, input_price_state, input_action], outputs=[output])
        model.compile(optimizer=Adam(self.LEARNING_RATE), loss='mse')
        plot_model(model, to_file='critic_net.png', show_shapes=True)
        return model, model.trainable_weights, input_stock_state, input_agent_state, input_price_state, input_action
