'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
###Default: BS256 LS128
#python3 tmp.py 256 20 10
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential,Model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Input,Flatten
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import random
import sys
import io
import os
import shutil


maxlen = 40
how_much_part = 1

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))


text = text[:int(len(text)/how_much_part)]
print('truncated corpus length:', len(text))
# print(text,file=open('../res/croped1_%d_nietzsche.txt' % how_much_part, 'a'))
# exit()

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


def word_split_out(text):
    word_list = []
    wcurrent = []

    for i, c in enumerate(text):
        if c.isalnum():
            wcurrent.append(c)
        elif wcurrent:
            word = u''.join(wcurrent)
            word_list.append(word)
            wcurrent = []

    if wcurrent:
        word = u''.join(wcurrent)
        word_list.append(word)

    return word_list

words = np.array(word_split_out(text))
len_of_each_words = np.array([len(w) for w in words])
max_len_of_words =  np.max(len_of_each_words)
# 20
np.average(max_len_of_words)
# 4.721024802827857

from scipy.stats import mode
mode(len_of_each_words)
print(mode(len_of_each_words)[0])
# [3]

def count_element_np(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result

count_ele = count_element_np(len_of_each_words)
print(count_ele)
#{1: 2571, 2: 20051, 3: 21368, 4: 15364, 5: 10075, 6: 7706, 7: 7452, 8: 5190, 9: 4591, 10: 3268, 11: 1784, 12: 1140, 13: 624, 14: 219, 15: 88, 16: 52, 17: 15, 18: 2, 20: 1}

np.std(len_of_each_words)
# 2.7165441669104182

import ipdb
ipdb.set_trace()
##Draw the accuracy picture
accuracy = []
current_right = 0.0
words_num = len(len_of_each_words)
max_len_of_words = int(np.max(len_of_each_words)) + 1
for i in range(1,max_len_of_words):
    current_right += count_ele[i]
    accuracy.append(current_right / words_num )
print(accuracy)

