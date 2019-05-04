#! /usr/bin/env python
#-*- coding:utf-8 -*-
import os
import codecs
import json
import  time
from utils import DATA_PROCESSED_DIR, uprintln
from segment import Segmenter
from quatrains import get_quatrains
# from vocab import get_vocab, VOCAB_SIZE
# import  tensorflow as tf
import  numpy as np
import jieba
from gensim import models
from random import shuffle, random, randint
from  functools import  reduce
import random
# from utils import uprintln, uprint, DATA_PROCESSED_DIR, split_sentences
# from data_utils import get_kw_train_data
# from segment import Segmenter
# from quatrains import get_quatrains
# from rank_words import get_word_ranks
# import pypinyin
# counters = dict()
# segmenter = Segmenter()  #generation   sxhy dict
# quatrains = get_quatrains()   #select poetry
# for idx, poem in enumerate(quatrains):
#     for sentence in poem['sentences']:
#         print(sentence)
# corpus=[]
# ch_lists=[]
# int2ch, ch2int = get_vocab()
# with codecs.open("./data/starterkit/qing.json","r","UTF-8")as fin:
#     data=json.load(fin)
#     corpus.extend(data)
#     for idx, poem in enumerate(corpus):
#         for sentence in poem['sentences']:
#             ch_lists.append(list(filter(lambda ch:ch in ch2int, sentence)))
#             print(sentence)
#             print(ch_lists)
#         time.sleep(1)
# # # A=dict()
# A["A"]=dict()
# A["A"]["B"]=123
# A["A"]["C"]=456
# print(A["A"].items())
# for other, weight in A["A"].items():
#     print(other)
# print(A["A"])
# for other in A["A"]:
#     print(other)
# with codecs.open("./data/starterkit/vocab.json","r","UTF-8")as fin:
#     data = json.load(fin)
#     print(data)
#     data=json.load(fin)
#     for idx, pair in enumerate(data):
#         print(idx,pair)vocab
# #         time.sleep(1)
# rows = []
# rows[0]=1
# kw_row = []
# for i in range(5):
#     kw_row.append(i)
#     rows[-1].append(i)
# print(kw_row,rows)
#
# import tensorflow as tf
#
# A = tf.Variable(tf.constant(0.0), dtype=tf.float32)
# B= tf.Variable(tf.constant(10.0), dtype=tf.float32)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#
#
#     print(sess.run(B.assign(A)))
#     print(A)

# import  numpy as np
#
# A=np.ones([2,2])
# print(A)
#
# B=np.ones([2,1])


# embedding_placeholder = tf.placeholder(
#             name='embedding_placeholder',
#             shape=[10, 5],
#             dtype=tf.float32
#         )
#
# embedding = tf.get_variable(
#     name='embedding',
#     shape=[10, 5],
#     trainable=False,
# )
#
# assign_embedding_op = embedding.assign(embedding_placeholder)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(assign_embedding_op))

# sentences="春风十里不如你"
# sentence=sentences[::-1]
# print(sentence)
# pad_len=3
# # result_len=5
# # for i in range(pad_len - result_len):  # 填充pad_token
# #     print(i)
# # print(4%4)
# # A=[1,2,3]
# # B=[4,5,6]
# # C=A+B
# # print(C)
# bidirectional=True

#array_ops.concat([inputs, state], 1)
# import os
#
# import numpy as np
# import sys
# from tensorflow.python.ops import array_ops
# a = tf.constant([[1,12,8,6], [3,4,6,7]])  # shape [2,4]
# b = tf.constant([[10, 20,6,88], [30,40,7,8]]) # shape [2,4]
# c = tf.constant([[10, 20,6,88,99], [30,40,7,8,15]]) #shape [2,5]
# d = tf.constant([[10,20,6,88], [30,40,7,8],[30,40,7,8]]) # shape [3,4]
# nn = tf.concat([a, d],0) # 按照第一维度相接，shape1 [a,m] shape2 [b,m] concat_done:[a+b,m]
# nn_1 = tf.concat([a, c],1) # 按照第二维度相接，shape1 [m,a] shape2 [m,b] concat_done:[m,a+b]
# mn = array_ops.concat([a, d], 1) # 按照第一维度相接，shape1 [a,m] shape2 [b,m] concat_done:[a+b,m]
# mn_1 = array_ops.concat([a, c], 1) # 按照第二维度相接，shape1 [m,a] shape2 [m,b] concat_done:[m,a+b]
#
# with tf.Session() as sess:
#      print (nn)
#      print (nn.eval())
#      print (nn_1)
#      print (nn_1.eval())
#      print (mn)
#      print (mn.eval())   # shape [5,4]
#      print (mn_1)
#      print (mn_1.eval()) # shape [2,9]
#
# test = np.array([[[1, 2, 3], [2, 3, 4]], [[5, 4, 3], [8, 7, 2]]])
# print(np.argmax(test, -1))
# data=[]
# data.append({'sentence': 1, 'keyword': 2})
# print(len(data))
# print(data)

# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# 11
# 12
# 13
# 14
# 15
# 16
# 17
# 18
# 19
# 20
# 21
# 22
# 23
# 24
# 25
# 26
# 27
# 28
# 29
# 30
# 31
# 32
# 33
# 34
# 35
# 36
# 37
# import numpy as np  # 这是Python的一种开源的数值计算扩展，非常强大
# import tensorflow as tf  # 导入tensorflow
#
# ##构造数据##
# x_data = np.random.rand(100).astype(np.float32)  # 随机生成100个类型为float32的值
# y_data = x_data * 0.1 + 0.3  # 定义方程式y=x_data*A+B
# ##-------##
#
# ##建立TensorFlow神经计算结构##
# weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# biases = tf.Variable(tf.zeros([1]))
# y = weight * x_data + biases
#
# w1 = weight * 2
#
# loss = tf.reduce_mean(tf.square(y - y_data))  # 判断与正确值的差距
# optimizer = tf.train.GradientDescentOptimizer(0.5)  # 根据差距进行反向传播修正参数
# train = optimizer.minimize(loss)  # 建立训练器
#
# init = tf.global_variables_initializer()  # 初始化TensorFlow训练结构
# # sess=tf.Session() #建立TensorFlow训练会话
# sess = tf.InteractiveSession()
# sess.run(init)  # 将训练结构装载到会话中
# print('weight', weight.eval())
# for step in range(400):  # 循环训练400次
#     sess.run(train)  # 使用训练器根据训练结构进行训练
#     if step % 20 == 0:  # 每20次打印一次训练结果
#         print(step, sess.run(weight), sess.run(biases))  # 训练次数，A值，B值
#
# print(sess.run(loss))
# print('weight new', weight.eval())
#
# wop=weight.assign([3])
# print("wop",wop.eval())
# #
# # weight.load([1], sess)
# # print('w1', w1.eval())
# previous_sentences_ints=[]
# current_sentence_ints=[3]
# previous_sentences_ints += [4] + current_sentence_ints
# print(previous_sentences_ints)
# current_sentence_ints=[5]
# previous_sentences_ints += [4] + current_sentence_ints
# print(previous_sentences_ints)
# def is_CN_char(ch):
#     return ch >= u'\u4e00' and ch <= u'\u9fa5'
#
#
# def split_sentences(line):
#     sentences = []
#     i = 0
#     for j in range(len(line)+1):
#         if j == len(line) or line[j] in [u'，', u'。', u'！', u'？', u'、']:
#             if i < j:
#                 sentence = u''.join(filter(is_CN_char, line[i:j]))
#                 print(sentence)
#                 sentences.append(sentence)
#             i = j+1
#     return sentences
# text="窗前明月光，疑是地上霜"
# print(split_sentences(text))
# ranks = get_word_ranks()
# quatrains = get_quatrains()
# segmenter = Segmenter()
# def extract(sentence):
#     return list(filter(lambda x: x in ranks, jieba.lcut(sentence)))
# # result=list(map(extract, split_sentences(text)))
# def get_ranks(line):
#     return ranks[line]
# input_dict=dict()
# for lines in result:
#     for line in lines:
#         input_dict[line]=ranks[line]
#         print("line:",line,"rank",ranks[line])
#
# result=sorted(result,key=get_ranks,reverse=False)
# print(result)
# keywords = sorted(reduce(lambda x, y: x + y, map(extract, split_sentences(text)), []),
#                   cmp=lambda x, y: cmp(self.ranks[x], self.ranks[y]))


# A=reduce(lambda x, y: x + y, map(extract, split_sentences(text)), [])
# [print(line,ranks[line])for line in A]
# keywords=sorted(A,key=get_ranks,reverse=True)
# print(A)
# print(keywords)
# print(keywords)
# # A=u'\uff0c'
# # print(A.encode("UTF-8"))
# print(1)
# for sentence in [] :
#     print(type(sentence))
#     print(2)
# print(3)\
# tone_rules = {
#     5: 3,
#     7: 4
# }
#
# print(tone_rules[7])
# def get_possible_tones( ch):
#     """
#     Args:
#         ch: A unicode character
#
#     Returns:
#         [int]: A list of possible tones
#
#     """
#     final_tones = pypinyin.pinyin(ch, style=pypinyin.FINALS_TONE3, heteronym=True, errors=u'default')[
#         0]  # select results for first and only char
#     tones = list(map(lambda final_tone: final_tone[-1], final_tones))
#     filtered_tones = list(filter(str.isdigit, tones))
#     tones_int = list(map(int, filtered_tones))
#
#     # deduplication
#     deduped_tones = []
#     for tone in tones_int:
#         if tone not in deduped_tones:
#             deduped_tones.append(tone)
#
#     return deduped_tones
# # print(get_possible_tones("一"))
# tones=[1]
# pin_tones = {1, 2} & set(tones)
# print(pin_tones)
# if  pin_tones:
#     print("春风十里不如你")

# 定义一个矩阵a，作为输入，也就是需要被圈记得矩阵

# a = np.array(np.arange(1, 41).reshape([2, 10, 2]), dtype=np.float32)
#
# print(a.shape)
#
# # 定义卷积核的大小，数目为1
#
# kernel = np.array(np.arange(1, 5), dtype=np.float32).reshape([2, 2, 1])
#
# print(a.shape)
#
# # 定义一个stride
#
# strides = 1
#
# # 进行conv1d卷积
#
# conv1d = tf.nn.conv1d(a, kernel, strides, 'VALID')
#
# with tf.Session() as sess:
#     # 初始化
#
#     tf.global_variables_initializer().run()
#
#     # 输出卷积值
#
#     print(sess.run(conv1d))
#
#     print(conv1d.shape)
random.seed(1)
a=np.random.randn(2,3,4)
b=a[-1]
print(b)
print(b[1])
b[1][0]=3
a[-1]=b
print(a)




