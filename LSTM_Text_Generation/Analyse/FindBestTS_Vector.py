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
# infor_length = int(sys.argv[1])

#infor_length is critical
def get_text_train_data(time_step = 10, infor_length = 15, how_much_part = 10):

    path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    with io.open(path, encoding='utf-8') as f:
        text = f.read().lower()
    # print('corpus length:', len(text))

    text = text[:int(len(text) / how_much_part)]
    # print('truncated corpus length:', len(text))

    chars = sorted(list(set(text)))
    # print('total chars:', len(chars))
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


#Hash Max
global_fail_hash_case = 0

class vector_pair:
    def get_hash(self, data):
        try:
            data.flags.writeable = False
            return hash(data.data)
        except:
            # import ipdb; ipdb.set_trace()
            return hash(str(data))
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
            acc = max(acc,self.labels[var])
        acc = acc / total_times
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


def run(length):
    train_data, train_label = get_text_train_data(infor_length=length)

    print('Build model...')

    text_pairs_hash = {}
    vector_pair_obj = vector_pair(1,2)
    for index, var in enumerate(train_data):
        hash_var = vector_pair_obj.get_hash(var)
        hash_label = vector_pair_obj.get_hash(train_label[index])
        if (hash_var in text_pairs_hash.keys()):
            text_pairs_hash[hash_var].add_label(hash_label)
        else:
            text_pairs_hash[hash_var] = vector_pair(hash_var, hash_label)
    print('Finish init!~')

    try:
        acc, count = calculate_res(text_pairs_hash)
        print(acc, count)
    except Exception as e:
        print(e)

    max_acc_log = './max_acc_vector_max.txt'
    # num / Acc
    print('%d \t %f' % (length, acc), file=open(max_acc_log, 'a'))
    # print('Fail hash case:', vector_pair.fail_hash_case)
    # import ipdb;ipdb.set_trace()

# for i in range(1,21):
#     run(i)

run(2)