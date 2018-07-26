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
import keras
import sys
import io
import os
import shutil

def get_text_train_data(time_step = 10, infor_length = 1, how_much_part = 10):

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
        sentences.append(text[i: i + infor_length])
        next_chars.append(text[i + infor_length])

    sentences_time_step = []
    next_chars_time_step = []
    for i in range(0, len(sentences) - time_step, step):
        sentences_time_step.append(sentences[i: i + time_step])
        next_chars_time_step.append(next_chars[i + time_step])

    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences_time_step), time_step, len(chars) * infor_length), dtype=np.bool)
    y = np.zeros((len(next_chars_time_step), len(chars)), dtype=np.bool)
    for i, sentences in enumerate(sentences_time_step):
        for t, senten in enumerate(sentences):
            for letterid, letter in enumerate(senten):
                # x[i, t, char_indices[letter]] = 1
                id_letter = char_indices[letter] + letterid * len(chars)
                x[i, t, id_letter] = 1
        id_ans_letter = char_indices[next_chars_time_step[i]]
        # print(id_ans_letter)
        y[i, id_ans_letter] = 1

    print(np.shape(x))
    print(np.shape(y))
    return x,y

train_data, train_label = get_text_train_data()

# build the model:  LSTM
print('Build model...')
