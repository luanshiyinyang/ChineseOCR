# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/8/8 16:18
   desc: predict image
"""
import os

import numpy as np
import cv2
import glob
from keras.layers import Input
from keras.models import Model
from keys import characters
import densenet
n_classes = len(characters)


def get_model():

    input_tensor = Input(shape=(32, None, 1), name='the_input')
    y_pred = densenet.dense_cnn(input_tensor, n_classes)
    base_model = Model(inputs=input_tensor, outputs=y_pred)
    model_file = '../models/best_model_weights.h5'
    if not os.path.exists(model_file):
        print("no model file")
    else:
        base_model.load_weights(model_file)

    return base_model


def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != n_classes - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])
    return ''.join(char_list)


def predict(img):
    """

    :param img: (H, W, 3) image
    :return:
    """
    target_height = 32
    height, width = img.shape[0], img.shape[1]
    # calculate scale factor
    scale = float(height / target_height)
    width = int(width / scale)
    # resize img
    img = cv2.resize(img, (width, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = np.array(img).astype(np.float32) / 255.0 - 0.5
    x = x.reshape([1, 32, width, 1])
    # get model and predict
    base_model = get_model()
    y_pred = base_model.predict(x)
    y_pred = y_pred[:, :, :]
    # decode pred vector
    out = decode(y_pred)
    print(out)
    return out


if __name__ == '__main__':
    images_file = glob.glob("../test_images/*")
    for file in images_file:
        image = cv2.imread(file)
        predict(image)
