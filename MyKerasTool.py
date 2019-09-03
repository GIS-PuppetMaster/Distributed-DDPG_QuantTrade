def Dense_BN(input, size=8):
    from keras.layers import Activation, BatchNormalization, Dense, Concatenate
    from keras import regularizers
    dense = Dense(size, kernel_regularizer=regularizers.l2(0.01))(input)
    dense = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense)
    return Activation('tanh')(dense)


def Dense_res_block3(input, layercell=(32, 16)):
    from keras.layers import Activation, BatchNormalization, Dense, Add
    from keras import regularizers
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input)
    ac = Activation('relu')(bn)
    dense01 = Dense(layercell[0], kernel_regularizer=regularizers.l2(0.01))(ac)
    dense01_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense01)
    dense01_ac = Activation('tanh')(dense01_bn)
    dense02 = Dense(layercell[1], kernel_regularizer=regularizers.l2(0.01))(dense01_ac)
    dense02_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense02)
    dense02_ac = Activation('tanh')(dense02_bn)
    dense03 = Dense(input.shape.as_list()[1], kernel_regularizer=regularizers.l2(0.01))(dense02_ac)
    merge = Add()([dense03, input])
    return merge


def Conv1D_res_block2(input, filters=(1), kernel_size=(3, 3), padding=('valid', 'valid'),
                      data_format=('channels_first', 'channels_first'), activation=('relu', 'relu'), zeropadding = True):
    from keras.layers import Conv1D, Activation, BatchNormalization, Concatenate, Add, ZeroPadding1D,Permute
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input)
    ac = Activation('relu')(bn)
    if zeropadding:
        ac = Permute((2, 1))(ac)
        ac = ZeroPadding1D(1)(ac)
        ac = Permute((2, 1))(ac)
    conv01 = ZeroPadding1D(1)(ac)
    conv01 = Conv1D(filters=filters[0], kernel_size=kernel_size[0], padding=padding[0], data_format=data_format[0])(
        conv01)
    conv01_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(conv01)
    conv01_ac = Activation(activation[0])(conv01_bn)
    if zeropadding:
        conv01_ac = Permute((2, 1))(conv01_ac)
        conv01_ac = ZeroPadding1D(1)(conv01_ac)
        conv01_ac = Permute((2, 1))(conv01_ac)
    conv02 = Conv1D(filters=input.shape.as_list()[1], kernel_size=kernel_size[1], padding=padding[1], data_format=data_format[1])(
        conv01_ac)
    merge02 = Add()([conv02, input])
    return merge02


def CuDNNLSTM_res_block2(input, size=(32)):
    from keras.layers import Conv1D, Activation, BatchNormalization, Concatenate, CuDNNLSTM, Bidirectional, Add
    from keras import regularizers
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input)
    ac = Activation('relu')(bn)
    lstm01 = Bidirectional(CuDNNLSTM(size[0], kernel_regularizer=regularizers.l2(0.01), return_sequences=True))(ac)
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(lstm01)
    ac = Activation('relu')(bn)
    lstm02 = CuDNNLSTM(input.shape.as_list()[2], kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(ac)
    merge = Add()([lstm02, input])
    return merge


def CuDNNLSTM_res_block3(input, size=(32, 32)):
    from keras.layers import Conv1D, Activation, BatchNormalization, Concatenate, CuDNNLSTM, Bidirectional, Add
    from keras import regularizers
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input)
    ac = Activation('relu')(bn)
    lstm01 = Bidirectional(CuDNNLSTM(size[0], kernel_regularizer=regularizers.l2(0.01), return_sequences=True))(ac)
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(lstm01)
    ac = Activation('relu')(bn)
    lstm02 = Bidirectional(CuDNNLSTM(size[1], kernel_regularizer=regularizers.l2(0.01), return_sequences=True))(ac)
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(lstm02)
    ac = Activation('relu')(bn)
    lstm03 = CuDNNLSTM(input.shape.as_list()[2], kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(ac)
    merge = Add()([lstm03, input])
    return merge


def Conv1D_res_block3(input, filters=(3, 3), kernel_size=(3, 3, 3), padding=('valid', 'valid', 'valid'),
                      data_format=('channels_first', 'channels_first', 'channels_first'),
                      activation=('relu', 'relu', 'relu'),
                      zeropadding=True):
    from keras.layers import Conv1D, Activation, BatchNormalization, Concatenate, Add, ZeroPadding1D, Permute
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input)
    ac = Activation('relu')(bn)
    if zeropadding:
        ac = Permute((2, 1))(ac)
        ac = ZeroPadding1D(1)(ac)
        ac = Permute((2, 1))(ac)
    conv01 = Conv1D(filters=filters[0], kernel_size=kernel_size[0], padding=padding[0], data_format=data_format[0])(ac)
    conv01_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(conv01)
    conv01_ac = Activation(activation[0])(conv01_bn)
    if zeropadding:
        conv01_ac = Permute((2, 1))(conv01_ac)
        conv01_ac = ZeroPadding1D(1)(conv01_ac)
        conv01_ac = Permute((2, 1))(conv01_ac)
    conv02 = Conv1D(filters=filters[1], kernel_size=kernel_size[1], padding=padding[1], data_format=data_format[1])(conv01_ac)
    conv02_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(conv02)
    conv02_ac = Activation(activation[1])(conv02_bn)
    if zeropadding:
        conv02_ac = Permute((2, 1))(conv02_ac)
        conv02_ac = ZeroPadding1D(1)(conv02_ac)
        conv02_ac = Permute((2, 1))(conv02_ac)
    conv03 = Conv1D(filters=input.shape.as_list()[1], kernel_size=kernel_size[2], padding=padding[2], data_format=data_format[2])(
        conv02_ac)
    merge03 = Add()([conv03, input])
    return merge03


def Dense_block_sparse(input, size=(8, 8, 8, 8, 8, 8, 8, 8, 8, 8)):
    from keras.layers import Concatenate
    dense00 = Dense_BN(input, size[0])
    dense01 = Dense_BN(input, size[1])
    dense10 = Dense_BN(dense00, size[2])
    dense11 = Dense_BN(dense01, size[3])
    dense20 = Concatenate()([dense10, dense00, dense01])
    dense20 = Dense_BN(dense20, size[4])
    dense21 = Concatenate()([dense00, dense01, dense11])
    dense21 = Dense_BN(dense21, size[5])
    dense30 = Concatenate()([dense10, dense20])
    dense30 = Dense_BN(dense30, size[6])
    dense31 = Concatenate()([dense21, dense11])
    dense31 = Dense_BN(dense31, size[7])
    dense40 = Concatenate()([dense30, dense20, dense21])
    dense40 = Dense_BN(dense40, size[8])
    dense41 = Concatenate()([dense20, dense21, dense31])
    dense41 = Dense_BN(dense41, size[9])
    return Concatenate()([dense40, dense41])
