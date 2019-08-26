# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/8/7 17:40
   desc: some keys about target
"""
characters_file = open('../data/char_std_all.txt', 'r', encoding='utf-8').readlines()  # 该文件包含大部分常用词
characters = ''.join([ch.strip('\n') for ch in characters_file][1:] + [' '])
