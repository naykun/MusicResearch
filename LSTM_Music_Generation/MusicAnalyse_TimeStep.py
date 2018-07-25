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
from sequence_example_lib import *
import os
import tensorflow as tf
sequence_example_file = ('~/sss/Mag/Mag_Data/data_from_WH/Transmit/S_E_BasicRNN_midishare_bach20180725/training_melodies.tfrecord')
sequence_example_file_paths = tf.gfile.Glob(
    os.path.expanduser(sequence_example_file))

inputs, labels, lengths = get_numpy_from_tf_sequence_example( input_size=38,
                                    sequence_example_file_paths = sequence_example_file_paths,
                                    shuffle = False)

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
    train_data, train_label = inputs, labels

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

for i in range(1,41):
    run(i)

# run(2)