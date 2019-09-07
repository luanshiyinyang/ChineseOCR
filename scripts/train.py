# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/8/7 17:28
   desc: train model in dataset
   出于比赛数据集中大部分图片已经进行了文本检测截取，这里直接DenseNet + CTC进行训练
"""
import os
from argparse import ArgumentParser
import warnings

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
import pandas as pd

from model import build_model
from keys import characters
from dataset import gen

model_file = '../models/best_model_weights.h5'

warnings.filterwarnings('ignore')
np.random.seed(2019)
IMG_HEIGHT = 32
IMG_WIDTH = 280
BATCH_SIZE = 128
MAX_LABEL_LENGTH = 10


def parse_command_params():
    """
    命令行参数解析器
    :return:
    """
    ap = ArgumentParser()  # 创建解析器
    ap.add_argument('-p', '--pretrained', default='no', help='if load pretrained model')
    ap.add_argument('-e', '--epochs', default='100', help='how many epochs to train')
    args_ = vars(ap.parse_args())
    return args_


def get_callback(model_path):
    """
    回调函数设定
    :return:
    """
    mc = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True,
                         save_weights_only=True, verbose=True)
    es = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    tb = TensorBoard(log_dir='../models/logs/', write_graph=True)
    return [mc, es, tb]


def get_dataset():
    df_all = pd.read_csv('../data/df_convert.csv', encoding='utf-8')
    df_all['filename'] = df_all['filename'].astype('str')
    df_all['text'] = df_all['text'].astype('str')
    # 取10000图片，按照8:2划分训练集验证集
    msk = np.random.rand(len(df_all)) < 0.8
    df_train = df_all[msk].reset_index(drop=True)
    df_test = df_all[~msk].reset_index(drop=True)
    train_generator = gen(df_train, '../data/train_images/', batchsize=BATCH_SIZE, maxlabellength=MAX_LABEL_LENGTH,
                          imagesize=(IMG_HEIGHT, IMG_WIDTH))
    test_generator = gen(df_test, '../data/train_images/', batchsize=BATCH_SIZE, maxlabellength=MAX_LABEL_LENGTH,
                         imagesize=(IMG_HEIGHT, IMG_WIDTH))
    return train_generator, test_generator, len(df_train), len(df_test)


def train():
    """
    训练
    :return:
    """
    # 获取命令行参数
    args = parse_command_params()
    # 构建模型
    n_classes = len(characters)
    basemodel, model = build_model(IMG_HEIGHT, n_classes)
    # 本地文件检查
    model_path = '../models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if os.path.exists(model_file) and args['pretrained'] == 'yes':
        basemodel.load_weights(model_file)
    # 获取训练数据
    train_gen, test_gen, n_train, n_test = get_dataset()
    # 训练，计算的是行准确率
    model.fit_generator(train_gen,
                        steps_per_epoch=n_train // BATCH_SIZE,
                        epochs=int(args['epochs']),
                        initial_epoch=0,
                        validation_data=test_gen,
                        validation_steps=n_test // BATCH_SIZE,
                        callbacks=get_callback(model_file))


if __name__ == '__main__':
    train()




