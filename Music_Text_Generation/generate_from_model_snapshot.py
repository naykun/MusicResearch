# coding: utf-8

# In[1]:


'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical
from keras.models import load_model

import re

import sys
import time
import numpy as np
import random
import os
from math import *

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import LambdaCallback

from my_to_midi import *

# In[ ]:


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_name', 'Wikifonia', 'Dataset name will be the prefix of exp_name')
tf.app.flags.DEFINE_string('dataset_dir', '/home/ouyangzhihao/sss/Mag/Mag_Data/TextMelody/Wikifonia/', 'Dataset Directory, which should contain name_train.txt and name_eval.txt')
tf.app.flags.DEFINE_integer('model_epoch', 150, 'To define which model to load')
tf.app.flags.DEFINE_string('model_dir',
                           '/unsullied/sharefs/ouyangzhihao/Share/LSTM/Text_Generation_Capacity/Code/MusicResearch/Music_Text_Generation/Model_logdir/', 'Model h5 directory')
tf.app.flags.DEFINE_string('exp_name', 'ResNet_8_16_firstConvWikifonia_batchS1024_epochs150_maxL64_2018-08-16_101617', 'Experiment name')
tf.app.flags.DEFINE_integer('generate_length', 1200, 'Number of steps of generated music')
tf.app.flags.DEFINE_integer('generate_num', 1, 'Number of steps of generated music')


# In[2]:



dataset_name = FLAGS.dataset_name
dataset_dir = FLAGS.dataset_dir
generate_length = FLAGS.generate_length
exp_name = FLAGS.exp_name
model_epoch = FLAGS.model_epoch
model_path = os.path.join(FLAGS.model_dir, exp_name, 'epoch%d.h5' % model_epoch)

# # exp_pat = r"([a-zA-Z]*)_batchS(\d*)_epochs(\d*)_units(\d*)_denseS(\d*)_maxL(\d*)_step(\d*)_embeddingL(\d*)_(.*)"
# exp_pat = r"(*_maxL(\d*)_step(\d*)_embeddingL(\d*)_(.*)"
# exp_mat = re.match(exp_pat, exp_name)
# exp_pat2 = r"([a-zA-Z]*)_batchS(\d*)_epochs(\d*)_units(\d*)_maxL(\d*)_step(\d*)_embeddingL(\d*)_(.*)"
# exp_mat2 = re.match(exp_pat2, exp_name)
#
# if exp_mat:
#     # batch_size = int(exp_mat.group(2))
#     # epochs = int(exp_mat.group(3))
#     # units = int(exp_mat.group(4))
#     # dense_size = int(exp_mat.group(5))
#     maxlen = int(exp_mat.group(0))
#     # step = int(exp_mat.group(7))
#     # embedding_length = int(exp_mat.group(8))
#     date_and_time = exp_mat.group(9)
# elif exp_mat2:
#     exp_mat = re.match(exp_pat2, exp_name)
#     batch_size = int(exp_mat.group(2))
#     epochs = int(exp_mat.group(3))
#     units = int(exp_mat.group(4))
#     maxlen = int(exp_mat.group(5))
#     step = int(exp_mat.group(6))
#     embedding_length = int(exp_mat.group(7))
#     date_and_time = exp_mat.group(8)
# else:
#     print("No match!")

maxlen = 64
embedding_length = 1
# In[6]:
generate_log_dir = os.path.join("Generate_logdir/", exp_name)
midi_log_dir = os.path.join(generate_log_dir, "midi")
text_log_dir = os.path.join(generate_log_dir, "text")
console_log_dir = os.path.join(generate_log_dir, "console")

new_date_and_time = time.strftime('%Y-%m-%d_%H%M%S')


def make_log_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


dirs = [generate_log_dir, midi_log_dir, text_log_dir, console_log_dir]
make_log_dirs(dirs)



def print_fn(str):
    print(str)
    console_log_file = os.path.join(console_log_dir, 'console_output.txt')
    with open(console_log_file, 'a+') as f:
        print(str, file=f)


print_fn("*"*20+exp_name+"*"*20)

# In[ ]:

train_dataset_path = os.path.join(dataset_dir, dataset_name+'_train.txt')
eval_dataset_path = os.path.join(dataset_dir, dataset_name+'_eval.txt')

with open(train_dataset_path, "r") as train_file:
    train_text = train_file.read().lower()
    train_file.close()

print_fn('Train dataset length: %d' % len(train_text))

with open(eval_dataset_path, "r") as eval_file:
    eval_text = eval_file.read().lower()
    eval_file.close()

print_fn('Eval dataset length: %d' % len(eval_text))

# In[3]:


chars = "._0123456789abcdefghijklmnopqrstuvwxyz"
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# wikifonia: 238w chars
# In[4]:

def text_to_events(str):
    ret = []
    for i in str:
        ret.append(char_indices[i])
    return ret



def get_embedded_data(text, maxlen, embedding_length):
    # cut the text in semi-redundant sequences of maxlen characters
    # inputs and labels are python strings

    inputs = [char_indices[var] for var in text]
    labels = [char_indices[var] for var in text]

    inputs = np.array(inputs)
    labels = np.array(labels)

    inputs = inputs[:len(inputs) - 1]
    labels = labels[1:len(labels)]

    inputs = to_categorical(inputs, len(chars))
    labels = to_categorical(labels, len(chars))

    inputs_emb = []
    label_emb = []
    for i in range(0, len(inputs) - embedding_length, 1):
        inputs_emb.append(inputs[i: i + embedding_length].flatten())
        label_emb.append(labels[i + embedding_length])

    inputs_maxlen = []
    label_maxlen = []
    for i in range(0, len(inputs_emb) - maxlen, 1):
        inputs_maxlen.append((inputs_emb[i: i + maxlen]))
        label_maxlen.append(label_emb[i+maxlen])

    # return inputs_emb, label_emb
    return np.array(inputs_maxlen), np.array(label_maxlen)


# In[7]:

print_fn('Vectorization...')
x_train, y_train = get_embedded_data(train_text, maxlen, embedding_length)
x_eval, y_eval = get_embedded_data(eval_text, maxlen, embedding_length)


# build the model: a single LSTM
print_fn('Load model...')
model = load_model(model_path)
model.summary()


# In[11]:
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    if temperature == 0:
        return np.argmax(preds)
    else:
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


# In[12]:


def generate_music(epoch, text, diversity, start_index, is_train=False):
    print_fn('----- diversity: %.1f' % diversity)

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print_fn('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(generate_length):
        x_pred = np.zeros((1, maxlen, len(chars) * embedding_length))
        for t, char in enumerate(sentence):
            for idx in range(embedding_length):
                x_pred[0, t, idx * embedding_length + char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

    if is_train:
        log_name = "epoch%d_train_diversity%02d" % (epoch, int(diversity * 10))
    else:
        if start_index == 0:
            log_name = "epoch%d_first_diversity%02d" % (epoch, int(diversity * 10))
        else:
            log_name = "epoch%d_random_diversity%02d" % (epoch, int(diversity * 10))

    log_name = new_date_and_time+"_"+log_name

    text_log_path = os.path.join(text_log_dir, log_name + ".txt")
    with open(text_log_path, "w") as text_log_file:
        text_log_file.write(generated + "\n")
        text_log_file.close()

    print_fn("Write %s.txt to %s" % (log_name, text_log_dir))

    events = text_to_events(generated)
    events_to_midi('basic_rnn', events, midi_log_dir, log_name)

    print_fn("Write %s.midi to %s" % (log_name, midi_log_dir))


def baseline_music(epoch, text, start_index, is_train=False):
    print_fn('----- baseline')

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence

    generated += eval_text[start_index + maxlen: min(len(text), start_index + maxlen + generate_length)]
    sys.stdout.write(generated)

    if is_train:
        log_name = "epoch%d_train_baseline" % epoch
    else:
        if start_index == 0:
            log_name = "epoch%d_first_baseline" % epoch
        else:
            log_name = "epoch%d_random_baseline" % epoch

    log_name = new_date_and_time+"_"+log_name

    text_log_path = os.path.join(text_log_dir, log_name + ".txt")
    with open(text_log_path, "w") as text_log_file:
        text_log_file.write(generated + "\n")
        text_log_file.close()

    print_fn("Write %s.txt to %s" % (log_name, text_log_dir))

    events = text_to_events(generated)
    events_to_midi('basic_rnn', events, midi_log_dir, log_name)

    print_fn("Write %s.midi to %s" % (log_name, midi_log_dir))


def generate(epoch):

    print_fn("")
    print_fn('----- Generating Music after Epoch: %d' % epoch)

    start_index = random.randint(0, len(eval_text) - maxlen - 1)

    # baseline_music(epoch=epoch, text=eval_text, start_index=0)
    baseline_music(epoch=epoch, text=eval_text, start_index=start_index)
    baseline_music(epoch=epoch, text=train_text, start_index=start_index, is_train=True)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        # generate_music(epoch=epoch, text=eval_text, diversity=diversity, start_index=0)
        generate_music(epoch=epoch, text=eval_text, diversity=diversity, start_index=start_index)
        generate_music(epoch=epoch, text=train_text, diversity=diversity, start_index=start_index, is_train=True)


# In[14]:
print_fn("*"*20+exp_name+"*"*20)
print_fn('x_train shape:'+str(np.shape(x_train)) )
print_fn('y_train shape:'+str(np.shape(y_train)) )

for i in range(FLAGS.generate_num):
    new_date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    generate(model_epoch)

'''
#generate_from_model_snapshot

rlaunch --cpu=4 --gpu=1 --memory=8000 bash


python3 g.py --dataset_name=Bach \
--dataset_dir=/home/ouyangzhihao/sss/Mag/Mag_Data/TextMelody/Bach/ \
--model_epoch=150 \
--model_dir=Model_logdir \
--exp_name=ResNet_8_16_firstConvWikifonia_batchS1024_epochs150_maxL64_2018-08-16_101617 \
--generate_length=1200 \
--generate_num=2


'''