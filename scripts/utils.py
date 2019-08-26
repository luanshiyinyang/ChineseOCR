# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/8/7 17:49
   desc: some util scripts
"""
from keys import characters
import pandas as pd


def one_hot(text, characters=characters):
    """
    实现字符映射的one-hot编码，不考虑字符数目上限，上限调控训练时处理
    :param text:
    :param characters:
    :return:
    """

    label = []
    for i, char in enumerate(text):
        index = characters.find(char)
        if index == -1:
            index = characters.find(u' ')
        label.append(str(index))

    return ' '.join(label)


def convert_df(df):
    """
    将DataFrame中text字段脱敏为字典中的下标
    :param df:
    :return:
    """
    import tqdm
    for i in tqdm.tqdm(range(len(df))):
        df.iloc[i, 3] = one_hot(df.iloc[i, 3], characters)


def create_vocab():
    """

    :return:
    """
    import tqdm
    chars = list(characters)
    annotion_file = "../data/train.list"
    df_annotion = pd.read_csv(annotion_file, sep='\t', names=['h', 'w', 'filename', 'text'])
    corpus = set()
    for i in tqdm.tqdm(range(len(df_annotion))):
        for item in str(df_annotion['text'][i]):
            if item not in characters:
                corpus.add(item)
    corpus = list(corpus)
    file = open('../data/char_std_all.txt', 'a', encoding='utf-8')
    for item in corpus:
        file.write('\n' + item)
    file.close()


if __name__ == "__main__":
    # ANNOTION_FILE = "../data/train.list"
    # df_annotion = pd.read_csv(ANNOTION_FILE, sep='\t', names=['h', 'w', 'filename', 'text'])
    # print(df_annotion.head())
    # convert_df(df_annotion)
    # print(df_annotion.head())
    create_vocab()
