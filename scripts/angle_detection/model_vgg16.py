# -*-coding:utf-8-*-
"""
Author: Zhou Chen
Date: 2019/9/6
Desc: construct model vgg16
"""
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten


def VGG16(input_shape=(224, 224, 3), n_classes=1000):
    """
    实现VGG16D的网络结构（著名的VGG16）
    没有使用Dropout和BN
    :param input_shape: the shape of input tensor
    :param n_classes: the number of classes
    :return: model object
    """
    # input layer
    input_tensor = Input(shape=input_shape)
    # block1
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    x = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block2
    x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block3
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block4
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block5
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # fc
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    output_tensor = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor, name='vgg16')
    return model


if __name__ == '__main__':
    vgg16 = VGG16()
    print(vgg16.summary())
