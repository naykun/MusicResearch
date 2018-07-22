'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
from __future__ import print_function
from keras.utils.data_utils import get_file
import numpy as np
import sys
import io
import os
import shutil

#Scan the infor_lenth to find the best timestep(eg. from 1 to 40)
infor_length = int(sys.argv[1])

#infor_length is critical
def get_text_train_data(time_step = 10, infor_length = 15, how_much_part = 10):

    path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    with io.open(path, encoding='utf-8') as f:
        text = f.read().lower()
    print('corpus length:', len(text))

    text = text[:int(len(text) / how_much_part)]
    print('truncated corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of time_step characters
    step = 1

    sentences = []
    next_chars = []
    for i in range(0, len(text) - infor_length, step):
        sentences.append(''.join(text[i: i + infor_length]))
        next_chars.append(text[i + infor_length])

    sentences_time_step = []
    next_chars_time_step = []
    for i in range(0, len(sentences) - time_step, step):
        sentences_time_step.append((sentences[i: i + time_step]))
        next_chars_time_step.append(next_chars[i + time_step])

    return sentences,next_chars

train_data, train_label = get_text_train_data(infor_length=infor_length)

print('Build model...')

class vector_pair:
    def __init__(self, input, label):
        self.labels = {}
        self.input = input
        self.add_label(label)
    def add_label(self,new_label):
        if not(new_label in self.labels):
            self.labels[new_label] = 1
        else:
            self.labels[new_label] += 1
    def get_acc(self):
        acc = 0.
        total_times = 0.
        for var in self.labels:
            total_times += self.labels[var]
            acc += self.labels[var]**2
        acc = acc / total_times**2
        return acc
    def get_total_times_in_dataset(self):
        total_times = 0
        for var in self.labels:
            total_times += self.labels[var]
        return total_times

def calculate_res(text_pairs):
    acc = 0
    res_count = 0.
    for vec in text_pairs:
        acc_text_p = text_pairs[vec].get_acc()
        count_text_p = text_pairs[vec].get_total_times_in_dataset()

        acc += acc_text_p * count_text_p
        res_count += count_text_p
    return acc / res_count , res_count


text_pairs = {}
for index, var in enumerate(train_data):
    if(var in text_pairs.keys()):
        text_pairs[var].add_label(train_label[index])
    else:
        text_pairs[var] = vector_pair(var, train_label[index])


print('Finish init!~')
try:
    acc, _ = calculate_res(text_pairs)
    print(acc, _)
except Exception as e:
    print(e)

max_acc_log = './max_acc.txt'
# num / Acc
print('%d \t %f' % (infor_length, acc), file=open(max_acc_log, 'a'))

# import ipdb; ipdb.set_trace()
# print(text_pairs)

# v1 = vector_pair('abcd','e')
# v2 = vector_pair('ab','f')
#
# text_pairs[v1.input] = v1
# text_pairs[v2.input] = v2

