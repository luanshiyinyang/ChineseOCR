# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/8/6 21:29
   desc: crnn to recognize text
"""
from keras.layers import Input
from keras.layers.core import Lambda
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

import densenet


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_model(img_h, n_classes):
    input_tensor = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.dense_blstm(input_tensor, n_classes)

    basemodel = Model(inputs=input_tensor, outputs=y_pred)

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda a, b: b}, optimizer=Adam(lr=3e-4), metrics=['accuracy'])

    return basemodel, model
