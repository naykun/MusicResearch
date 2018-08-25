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
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import random
import sys
import io
import os
import shutil
from ConvModel import *

if(len(sys.argv)!=1):
    maxlen = int(sys.argv[2])
    how_much_part = int(sys.argv[3])
else:
    maxlen = 32
    how_much_part = 10

epochs = 50
batch_size = 1024

def lr_schedule(epoch):
    #Learning Rate Schedule
    lr = 1e-3
    if epoch >= epochs * 0.9:
        lr *= 0.5e-3
    elif epoch >= epochs * 0.8:
        lr *= 1e-3
    elif epoch >= epochs * 0.6:
        lr *= 1e-2
    elif epoch >= epochs * 0.4:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

exp_name = 'Debug_WinSize%d_BS%d_%dpart_epoch%d' % (maxlen, batch_size, how_much_part, epochs)
log = '../res/' + exp_name + '.txt'
tb_log = '../TB_logdir/CNN/' + exp_name
max_acc_log = '../res/max_acc.txt'
model_log_dir = '../Models/' + exp_name
initial_epoch = 0

if(os.path.exists(os.path.join(model_log_dir, '%d.h5' % epochs))):
    print('Already Finished Training')
    # exit()

if not(os.path.exists('../res/')):os.mkdir('../res/')
if not(os.path.exists('../Models/')):os.mkdir('../Models/')
# if(os.path.exists(log)):os.remove(log)
# if(os.path.exists(tb_log)):shutil.rmtree(tb_log)


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

print('x shape ', x.shape)
print('y shape ',y.shape)

# build the model:  LSTM
print('Build model...')
train_input_shape = (maxlen, len(chars))
train_output_shape = (len(chars))

model = get_conv1d_model(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_conv1d_model_old(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_two_pipeline_model(input_shape=train_input_shape,output_shape = train_output_shape)

# model = get_naive_conv1d_model(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_conv1d_model_simple(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_resNet_model(input_shape=train_input_shape,output_shape = train_output_shape)

# model = get_lstm_model(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_conv1d_resnet(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_complex_model(input_shape_melody=train_input_shape, input_shape_accom=train_input_shape,  output_shape=train_output_shape)
# model = get_complex_model_resNet_melody(input_shape_melody=train_input_shape, input_shape_accom=train_input_shape,  output_shape=train_output_shape)

model.summary()
optimizer = Adam(lr=lr_schedule(0))

from keras import metrics

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy','ce'])

#Continue train
if(os.path.exists(model_log_dir)):
    if(len(os.listdir(model_log_dir)) > 0 ):
        file_name = os.listdir(model_log_dir)[0]
        file_path = os.path.join(model_log_dir, file_name)
        # model = keras.models.load_model(file_path)
        # initial_epoch = int(file_name.split('.')[0])
        # print('load old model: %s \n with epoch %d' % (model_log_dir, initial_epoch))
else:
    os.mkdir(model_log_dir)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def print_func(str = '\n'):
    print(str)
    print(str, file=open(log, 'a'))

def on_epoch_end(epoch, logs):
    return
    if(epoch%10 == 0):
        shutil.rmtree(model_log_dir)
        os.mkdir(model_log_dir)
        model_name = os.path.join(model_log_dir, '%d.h5' % epoch)
        model.save(model_name)
    if((epoch+1)%20  == 0):
        # Function invoked at end of each epoch. Prints generated text.
        print_func()
        print_func('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print_func('----- diversity:' + str(diversity))

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print_func('----- Generating with seed: "' + sentence + '"')

            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()

            print_func(generated)
            print_func()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
lr_scheduler = LearningRateScheduler(lr_schedule)
# 参照下面代码加一下TensorBoard
from keras.callbacks import TensorBoard,ReduceLROnPlateau
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=50,
                               min_lr=0.5e-6)

checkpoint = ModelCheckpoint(filepath=model_log_dir + 'test.h5',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

from keras import backend as K
from keras.callbacks import TensorBoard

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

tb_callbacks = LRTensorBoard(log_dir = tb_log)


history_callback = model.fit(x, y, validation_split=0.1,
          verbose = 1,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[print_callback,tb_callbacks,lr_scheduler],initial_epoch = initial_epoch)


acc_history = history_callback.history["acc"]

max_acc = str(np.max(acc_history))
print('Exp max accuracy:',max_acc)
print(exp_name  + '\t' + str(maxlen) + '\t' + max_acc, file=open(max_acc_log, 'a'))

shutil.rmtree(model_log_dir)
os.mkdir(model_log_dir)
model_name = os.path.join(model_log_dir, '%d.h5' % epochs)
model.save(model_name)