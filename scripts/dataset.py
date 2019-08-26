# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/8/7 17:33
   desc: load data from folder
"""
import cv2
import numpy as np


def readfile(filename):
    """
    要求文件为文本文件且每行放置多个字符串用空格间隔，第一个字符串为图片文件名，后面的为脱敏单词索引号
    :param filename:
    :return:
    """
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic


def read_df(df):
    """
    类似上面的文本文件，不过文件名在filename列下，脱敏索引在text列下
    :param df:
    :return:
    """
    dic = dict()
    for i in range(len(df)):
        dic[df['filename'][i]] = df['text'][i].strip().split(' ')
    return dic


class RandomUniformNum(object):
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batchsize):
        r_n = []
        if self.index + batchsize > self.total:
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index : self.index + batchsize]
            self.index = self.index + batchsize

        return r_n


def gen(data_file, image_path, batchsize=128, maxlabellength=10, imagesize=(32, 280)):
    image_label = read_df(data_file)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = RandomUniformNum(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while True:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            # 取图片
            img = cv2.imread(image_path+j)
            img = cv2.resize(img, (imagesize[1], imagesize[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.array(img, 'float32') / 255.0 - 0.5
            x[i] = np.expand_dims(img, axis=-1)  # 在末尾添加一个空维度
            # 取文本
            text = image_label[j]
            label_length[i] = min(maxlabellength, len(text))

            if len(text) <= 0:
                print("file {}, text length < 0".format(j))
            input_length[i] = imagesize[1] // 8
            # 非负避免负映射
            labels[i, :min(maxlabellength, len(text))] = [int(k)+1 for k in text][:min(maxlabellength, len(text))]

        inputs = {'the_input': x,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)
