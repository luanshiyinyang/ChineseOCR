# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/8/7 18:31
   desc: implement densenet structure
"""

from keras.layers.core import Dense, Dropout, Activation, Permute
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Flatten
from keras.layers import Bidirectional, LSTM
from keras.regularizers import l2


def conv_block(x, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, dropout_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter


def transition_block(x, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    if pooltype == 2:
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif pooltype == 1:
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif pooltype == 3:
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter


def dense_cnn(x, n_classes):
    """
    使用DenseNet进行提取
    :param x:
    :param n_classes:
    :return:
    """
    dropout_rate = 0.2
    weight_decay = 1e-4

    nb_filter = 64
    # conv 64 5*5 s=2
    x = Conv2D(nb_filter, (5, 5), strides=(2, 2),
               kernel_initializer='he_normal',
               padding='same',
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    # 64 + 8 * 8 = 128
    x, _nb_filter = dense_block(x, 8, nb_filter, 8, None, weight_decay)
    # 128
    x, _nb_filter = transition_block(x, 128, dropout_rate, 2, weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, weight_decay)
    # 192 -> 128
    x, _nb_filter = transition_block(x, 128, dropout_rate, 2, weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = Permute((2, 1, 3), name='permute')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)
    y_pred = Dense(n_classes, name='out', activation='softmax')(x)

    # basemodel = Model(inputs=input, outputs=y_pred)
    # basemodel.summary()

    return y_pred


def dense_blstm(x, n_classes):
    """
    加入BLSTM进行前部模型的设计
    :param x:
    :param n_classes:
    :return:
    """
    dropout_rate = 0.2
    weight_decay = 1e-4

    nb_filter = 64
    # conv 64 5*5 s=2
    x = Conv2D(nb_filter, (5, 5), strides=(2, 2),
               kernel_initializer='he_normal',
               padding='same',
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    # 64 + 8 * 8 = 128
    x, _nb_filter = dense_block(x, 8, nb_filter, 8, None, weight_decay)
    # 128
    x, _nb_filter = transition_block(x, 128, dropout_rate, 2, weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, weight_decay)
    # 192 -> 128
    x, _nb_filter = transition_block(x, 128, dropout_rate, 2, weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = Permute((2, 1, 3), name='permute')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)

    rnnunit = 256
    x = Bidirectional(LSTM(rnnunit, return_sequences=True, implementation=2), name='blstm1')(x)
    x = Dense(rnnunit, name='blstm1_out', activation='linear')(x)
    x = Bidirectional(LSTM(rnnunit, return_sequences=True, implementation=2), name='blstm2')(x)
    y_pred = Dense(n_classes, name='out', activation='softmax')(x)

    return y_pred


if __name__ == '__main__':
    input = Input(shape=(32, 280, 1), name='the_input')
    dense_cnn(input, 5000)