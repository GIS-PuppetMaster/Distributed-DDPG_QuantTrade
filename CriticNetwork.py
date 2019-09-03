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
        from MyKerasTool import Dense_block_sparse
        from MyKerasTool import Conv1D_res_block2
        from MyKerasTool import Conv1D_res_block3
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
        input_agent_state_ = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input_agent_state)
        input_price_state = Input(shape=(glo.price_state_size,))
        input_action = Input(shape=(glo.action_size,))
        x_stock_state = SeparableConv1D(filters=64, kernel_size=4, padding='valid', data_format='channels_first')(
            input_stock_state)
        x_stock_state = BatchNormalization(epsilon=1e-4, scale=True, center=True)(x_stock_state)
        x_stock_state = Activation('tanh')(x_stock_state)
        x_stock_state = SeparableConv1D(filters=32, kernel_size=3, padding='valid', data_format='channels_first')(
            x_stock_state)
        x_stock_state = BatchNormalization(epsilon=1e-4, scale=True, center=True)(x_stock_state)
        x_stock_state = Activation('tanh')(x_stock_state)
        x_stock_state = CuDNNLSTM_res_block3(x_stock_state, (16, 32))
        x_stock_state = CuDNNLSTM_res_block3(x_stock_state, (8, 16))
        x_stock_state = CuDNNLSTM_res_block3(x_stock_state, (8, 8))
        x_stock_state = CuDNNLSTM_res_block3(x_stock_state, (8, 8))
        x_stock_state = CuDNNLSTM_res_block3(x_stock_state, (32, 8))
        x_stock_state = BatchNormalization(epsilon=1e-4, scale=True, center=True)(x_stock_state)
        x_stock_state = Activation('tanh')(x_stock_state)
        x_stock_state = SeparableConv1D(filters=16, kernel_size=3, padding='valid', data_format='channels_first')(
            x_stock_state)
        x_stock_state = BatchNormalization(epsilon=1e-4, scale=True, center=True)(x_stock_state)
        x_stock_state = Activation('tanh')(x_stock_state)
        x_stock_state = CuDNNLSTM_res_block2(x_stock_state, (16,))
        x_stock_state = SeparableConv1D(filters=16, kernel_size=1, padding='valid', data_format='channels_first')(
            x_stock_state)
        x_stock_state = BatchNormalization(epsilon=1e-4, scale=True, center=True)(x_stock_state)
        x_stock_state = Activation('tanh')(x_stock_state)
        feature_stock_state = Conv1D_res_block3(x_stock_state, (16, 4), (3, 3, 3))
        feature_stock_state = Conv1D_res_block3(feature_stock_state, (4, 4), (1, 1, 1), zeropadding=False)
        feature_stock_state = Conv1D_res_block3(feature_stock_state, (4, 4), (3, 3, 3))
        # feature_stock_state = Flatten()(feature_stock_state)
        feature_stock_state = AveragePooling1D(pool_size=2, strides=16, padding='valid', data_format='channels_last')(
            feature_stock_state)
        feature_stock_state = Reshape((12,))(feature_stock_state)
        feature_stock_state = Dense_res_block3(feature_stock_state, (8, 8))
        feature_stock_state = Dense_res_block3(feature_stock_state, (16, 8))
        feature_stock_state = Dense_res_block3(feature_stock_state, (16, 8))
        merge_layer = Concatenate()([feature_stock_state, input_agent_state_, input_price_state, input_action])
        analysis = Dense_res_block3(merge_layer, layercell=(16, 16))
        analysis = Dense_res_block3(analysis, layercell=(16, 8))
        analysis = Dense_res_block3(analysis, layercell=(8, 8))
        analysis = Dense_res_block3(analysis, layercell=(8, 8))
        analysis = BatchNormalization(epsilon=1e-4, scale=True, center=True)(analysis)
        analysis = Activation('tanh')(analysis)
        output = Dense(1, kernel_regularizer=regularizers.l2(0.01))(analysis)
        model = Model(inputs=[input_stock_state, input_agent_state, input_price_state, input_action], outputs=[output])
        model.compile(optimizer=Adam(glo.critic_learning_rate), loss='mse')
        plot_model(model, to_file='critic_net.png', show_shapes=True)
        return model, model.trainable_weights, input_stock_state, input_agent_state, input_price_state, input_action
