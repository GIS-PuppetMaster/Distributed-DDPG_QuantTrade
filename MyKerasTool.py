def Dense_BN(input, units=8, bn_axis=-1):
    from keras.layers import Activation, BatchNormalization, Dense, Concatenate
    from keras import regularizers
    dense = Dense(units, kernel_regularizer=regularizers.l2(0.01))(input)
    dense = BatchNormalization(axis=bn_axis, epsilon=1e-4, scale=True, center=True)(dense)
    return Activation('tanh')(dense)


def Dense_res_block3(input, layercell=(32, 16)):
    from keras.layers import Activation, BatchNormalization, Dense, Add
    from keras import regularizers
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input)
    ac = Activation('tanh')(bn)
    dense01 = Dense(layercell[0], kernel_regularizer=regularizers.l2(0.01))(ac)
    dense01_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense01)
    dense01_ac = Activation('tanh')(dense01_bn)
    dense02 = Dense(layercell[1], kernel_regularizer=regularizers.l2(0.01))(dense01_ac)
    dense02_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense02)
    dense02_ac = Activation('tanh')(dense02_bn)
    dense03 = Dense(input.shape.as_list()[1], kernel_regularizer=regularizers.l2(0.01))(dense02_ac)
    merge = Add()([dense03, input])
    return merge


def CuDNNLSTM_res_block2(input, size=(32)):
    from keras.layers import Conv1D, Activation, BatchNormalization, Concatenate, CuDNNLSTM, Bidirectional, Add
    from keras import regularizers
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input)
    ac = Activation('tanh')(bn)
    lstm01 = CuDNNLSTM(size[0], kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(ac)
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(lstm01)
    ac = Activation('tanh')(bn)
    lstm02 = CuDNNLSTM(input.shape.as_list()[2], kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(ac)
    merge = Add()([lstm02, input])
    return merge


def CuDNNLSTM_res_block3(input, size=(32, 32)):
    from keras.layers import Conv1D, Activation, BatchNormalization, Concatenate, CuDNNLSTM, Bidirectional, Add
    from keras import regularizers
    lstm01 = CuDNNLSTM(size[0], kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(input)
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(lstm01)
    ac = Activation('tanh')(bn)
    lstm02 = CuDNNLSTM(size[1], kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(ac)
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(lstm02)
    ac = Activation('tanh')(bn)
    lstm03 = CuDNNLSTM(input.shape.as_list()[2], kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(ac)
    merge = Add()([lstm03, input])
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(merge)
    ac = Activation('tanh')(bn)
    return ac


def Conv1D_identity_block(input, filters=(3, 3, 3), kernel_size=3,
                      padding=('valid', 'same', 'valid'),
                      data_format='channels_first',
                      activation=('tanh', 'tanh', 'tanh'), block_name='Conv1D_identity_block'):
    from keras.layers import Conv1D, Activation, BatchNormalization, Concatenate, Add, ZeroPadding1D, Permute
    shortcut = input
    conv01 = Conv1D(filters=filters[0], kernel_size=1, padding=padding[0], strides=1, data_format=data_format, name=block_name+'conv01')(input)
    conv01_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'conv01_bn')(conv01)
    conv01_ac = Activation(activation[0], name=block_name+'conv01_ac')(conv01_bn)

    conv02 = Conv1D(filters=filters[1], kernel_size=kernel_size, padding=padding[1], strides=1, data_format=data_format, name=block_name+'conv02')(conv01_ac)
    conv02_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'conv02_bn')(conv02)
    conv02_ac = Activation(activation[1], name=block_name+'conv02_ac')(conv02_bn)

    conv03 = Conv1D(filters=filters[2], kernel_size=1, padding=padding[2], strides=1, data_format=data_format, name=block_name+'conv03')(conv02_ac)
    conv03_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'conv03_bn')(conv03)
    merge03 = Add(name=block_name+'merge03_Add')([conv03_bn, shortcut])
    conv03_ac = Activation(activation[2], name=block_name+'conv03_ac')(merge03)
    return conv03_ac


def Conv1D_conv_block(input, filters=(3, 3, 3), kernel_size=3, strides=1,
                      padding=('valid', 'same', 'valid', 'valid'),
                      data_format='channels_first',
                      activation=('tanh', 'tanh', 'tanh'), block_name='Conv1D_conv_block'):
    from keras.layers import Conv1D, Activation, BatchNormalization, Concatenate, Add, ZeroPadding1D, Permute
    shortcut = input
    conv01 = Conv1D(filters=filters[0], kernel_size=1, padding=padding[0], strides=strides, data_format=data_format, name=block_name+'conv01')(input)
    conv01_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'conv01_bn')(conv01)
    conv01_ac = Activation(activation[0], name=block_name+'conv01_ac')(conv01_bn)

    conv02 = Conv1D(filters=filters[1], kernel_size=kernel_size, padding=padding[1], strides=1,
                    data_format=data_format, name=block_name+'conv02')(conv01_ac)
    conv02_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'conv02_bn')(conv02)
    conv02_ac = Activation(activation[1], name=block_name+'conv02_ac')(conv02_bn)

    conv03 = Conv1D(filters=filters[2], kernel_size=1, padding=padding[2], strides=1, data_format=data_format, name=block_name+'conv03')(conv02_ac)
    conv03_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'conv03_bn')(conv03)

    shortcut = Conv1D(filters=filters[2], kernel_size=1, padding=padding[3], strides=strides, data_format=data_format, name=block_name+'shortcut_conv')(shortcut)
    shortcut = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'shortcut_bn')(shortcut)
    merge03 = Add(name=block_name+'merge03_Add')([conv03_bn, shortcut])
    conv03_ac = Activation(activation[2], name=block_name+'conv03_ac')(merge03)
    return conv03_ac

def Dense_layer_connect(input, size, units=8):
    from keras.layers import Concatenate,Dense,BatchNormalization,Activation,Flatten,Reshape
    from keras import regularizers
    # 升维，增加深度轴
    input_ = Reshape((1, size))(input)
    dense0 = Dense(units, kernel_regularizer=regularizers.l2(0.01))(input_)
    dense0_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense0)
    dense0 = Activation('tanh')(dense0_bn)

    dense1 = Dense(units, kernel_regularizer=regularizers.l2(0.01))(dense0)
    dense1_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense1)
    dense1 = Activation('tanh')(dense1_bn)

    dense2 = Dense(units, kernel_regularizer=regularizers.l2(0.01))(dense1)
    dense2_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense2)
    # 在深度轴上合并
    add2 = Concatenate(axis=1)([dense0_bn, dense2_bn])
    dense2 = Activation('tanh')(add2)

    dense3 = Dense(units, kernel_regularizer=regularizers.l2(0.01))(dense2)
    dense3_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense3)
    add3 = Concatenate(axis=1)([dense0_bn, dense1_bn, dense3_bn])
    dense3 = Activation('tanh')(add3)
    # 降维展平
    flatten = Flatten()(dense3)
    return flatten
