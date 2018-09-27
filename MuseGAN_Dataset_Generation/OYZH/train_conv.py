from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical


import time, copy
import numpy as np
import random
import sys
import os
from math import *

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import LambdaCallback

import my_config
from my_midi_io import *
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('maxlen', 96, '')
tf.app.flags.DEFINE_integer('generate_length', 1000, '')
tf.app.flags.DEFINE_integer('step', 8, '')
tf.app.flags.DEFINE_integer('batch_size', 256, '')
tf.app.flags.DEFINE_integer('epochs', 150, '')


# set up config
config = copy.deepcopy(my_config.config_5b)
dataset = np.load(config['dataset_path'])
print(config)


# In[3]:

# load dataset
dataset = dataset[0:100]
print("dataset shape:", dataset.shape)

maxlen = FLAGS.maxlen
generate_length = FLAGS.generate_length
step = FLAGS.step
epochs = FLAGS.epochs
batch_size = FLAGS.batch_size

# feature
reshaped_dataset = dataset.reshape((len(dataset), -1, 84, 5))

print('reshaped_dataset', reshaped_dataset.shape)



feature = []
label = []

for now_song in reshaped_dataset:
    for i in range(0, len(now_song), step):
        if (i + maxlen + 1) < len(now_song):
            # print('now_song', now_song.shape)
            feature.append(now_song[i:i+maxlen])
            label.append(now_song[i+maxlen])
            # print('feature', np.shape(feature))

print('reshaped_dataset', np.shape(feature))
# import ipdb; ipdb.set_trace()

# split data to train and validation
feature_size = len(feature)
val_ratio = 0.1

train_size = int(feature_size * (1 - val_ratio))

x_train = np.array(feature[0:train_size])
y_train = np.array(label[0:train_size])

x_val = np.array(feature[train_size: feature_size])
y_val = np.array(label[train_size: feature_size])


# In[9]:


print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_val shape:', x_val.shape)
print('y_val shape:', y_val.shape)



from ConvModel import *
from ConvResModel import *
from keras import backend as K


# model = get_conv1d_model_naive(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_conv1d_model(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_resnet_model_naive(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_resNet_model(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_lstm_model(input_shape=train_input_shape,output_shape = train_output_shape)
# model = resnet_v1_110(input_shape=train_input_shape,output_shape = train_output_shape)
# get_model_fn = get_conv1d_model_b_small

exp_name = 'a_MuseGan_KL'
# get_model_fn = get_conv1d_model_naive_big
get_model_fn = get_conv1d_model_a
# get_model_fn = get_conv1d_model_naive

# get_model_fn = get_conv1d_model

vector_dim = (84,5)
# import ipdb; ipdb.set_trace()
# print(train_data[0].shape)
# print(train_data[0])


# train_data = train_data[0:10000]
# eval_data = eval_data[0:2000]

# In[6]:
log_root = '/unsullied/sharefs/ouyangzhihao/Share/LSTM/Text_Generation_Capacity/Code/Music_Research_Exp/Music_Text_Gneration/9_19/MuseGan/'
log_dir = os.path.join(log_root, "logdir", exp_name)
TB_log_dir = os.path.join(log_root, 'TB_logdir', exp_name)
console_log_dir = os.path.join(log_root, log_dir, "console")
model_log_dir = os.path.join(log_root, 'Model_logdir', exp_name)
data_log_dir = os.path.join(log_root, log_dir, "data")
midi_log_dir = os.path.join(log_root, log_dir, "midi")


def make_log_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


dirs = [log_dir, TB_log_dir, console_log_dir, model_log_dir, data_log_dir, midi_log_dir]
make_log_dirs(dirs)

max_acc_log_path = os.path.join(log_root, "logdir", "max_acc_log.txt")


def get_embedded_data(data, maxlen, embedding_length):
    # cut the data in semi-redundant sequences of maxlen characters

    inputs = data[:len(data) - 1]
    labels = data[1:len(data)]

    print(np.shape(inputs))
    print(np.shape(inputs[0]))
    print((inputs[0]))

    inputs = to_categorical(inputs, 259)
    labels = to_categorical(labels, 259)

    inputs_emb = []
    label_emb = []
    for i in range(0, len(inputs) - embedding_length, 1):
        inputs_emb.append(inputs[i: i + embedding_length].flatten())
        label_emb.append(labels[i + embedding_length])

    inputs_maxlen = []
    label_maxlen = []
    for i in range(0, len(inputs_emb) - maxlen, step):
        inputs_maxlen.append((inputs_emb[i: i + maxlen]))
        label_maxlen.append(label_emb[i+maxlen])

    # return inputs_emb, label_emb
    return np.asarray(inputs_maxlen, dtype=np.float16), np.asarray(label_maxlen,dtype=np.float16)
# In[8]:


def print_fn(str):
    print(str)
    console_log_file = os.path.join(console_log_dir, 'console_output.txt')
    with open(console_log_file, 'a+') as f:
        print(str, file=f)

def lr_schedule(epoch):
    # Learning Rate Schedule

    lr = 1e-2
    if epoch >= epochs * 0.9:
        lr *= 0.5e-3
    elif epoch >= epochs * 0.8:
        lr *= 1e-3
    elif epoch >= epochs * 0.6:
        lr *= 1e-2
    elif epoch >= epochs * 0.4:
        lr *= 1e-1
    print_fn('Learning rate: %f' % lr)

    lr = 1e-3
    return lr


print_fn('Build model...')


train_output_shape = vector_dim
train_input_shape = ( maxlen, vector_dim[0], vector_dim[1] )

model = get_model_fn(input_shape=train_input_shape,output_shape = train_output_shape, timestep=maxlen)

def perplexity(y_trues, y_preds):
    cross_entropy = K.categorical_crossentropy(y_trues, y_preds)
    perplexity = K.pow(2.0, cross_entropy)

    #Another Method
    # oneoverlog2 = 1.442695
    # result = K.log(x) * oneoverlog2
    return perplexity

optimizer = Adam(lr=lr_schedule(0))
# model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy',perplexity])
model.compile(loss='kullback_leibler_divergence', optimizer=optimizer, metrics=['accuracy',perplexity])


model.summary(print_fn=print_fn)



def generate_music(epoch, data, diversity, start_index, is_train=False):
    print_fn('----- diversity: %.1f' % diversity)

    generated = [0]
    events = data[start_index: start_index + maxlen]
    generated += events
    print('----- Generating with seed: ', events)
    print(generated)

    generated = list(generated)

    for i in range(generate_length):
        x_pred = np.zeros((1, maxlen, 259 * embedding_length))
        for t, event in enumerate(events):
        #     # for idx in range(embedding_length):
        #
        #     print("debug:", t, event % 259)
        #
            x_pred[0, t, event % 259] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_event = int(next_index)

        generated.append(next_event)
        events = events[1:] + [next_event]

        # print(next_event, end=',')

    # print('')

    if is_train:
        log_name = "epoch%d_train_diversity%02d" % (epoch + 1, int(diversity * 10))
    else:
        if start_index == 0:
            log_name = "epoch%d_first_diversity%02d" % (epoch + 1, int(diversity * 10))
        else:
            log_name = "epoch%d_random_diversity%02d" % (epoch + 1, int(diversity * 10))


    # generated = list(generated)
    generated += [1]

    data_log_path = os.path.join(data_log_dir, log_name + ".pkl")
    with open(data_log_path, "wb") as data_log_file:
        data_log_file.write(pkl.dumps(generated) )
        data_log_file.close()

    print_fn("Write %s.pkl to %s" % (log_name, data_log_dir))

    list_to_midi(generated, 120, midi_log_dir, log_name)

    print_fn("Write %s.midi to %s" % (log_name, midi_log_dir))

    model_name = "epoch%d.h5" % (epoch+1)
    model_path = os.path.join(model_log_dir, model_name)
    model.save(model_path)
    print_fn("Save model %s.h5 to %s" % (model_name, model_log_dir))


def generate_more_midi(id, data, diversity, start_index, eval_input=False):
    print_fn('----- diversity: %.1f' % diversity)

    generated = [0]
    events = data[start_index: start_index + maxlen]
    generated += events
    print('----- Generating with seed: ', events)
    print(generated)

    generated = list(generated)
    # generate_length = 3200
    for i in range(generate_length):
        x_pred = np.zeros((1, maxlen, 259 * embedding_length))
        for t, event in enumerate(events):
            x_pred[0, t, event % 259] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_event = int(next_index)

        generated.append(next_event)
        events = events[1:] + [next_event]

    # print('')

    if eval_input:
        log_name = "%d_eval_diversity%02d" % (id + 1, int(diversity * 10))
    else:
        log_name = "%d_train_diversity%02d" % (id + 1, int(diversity * 10))


    # generated = list(generated)
    generated += [1]

    data_log_path = os.path.join(data_log_dir, log_name + ".pkl")
    with open(data_log_path, "wb") as data_log_file:
        data_log_file.write(pkl.dumps(generated) )
        data_log_file.close()

    print_fn("Write %s.pkl to %s" % (log_name, data_log_dir))

    list_to_midi(generated, 120, midi_log_dir, log_name)

    print_fn("Write %s.midi to %s" % (log_name, midi_log_dir))


converage_epoch = -1
def on_epoch_end(epoch, logs):
    #OYZH
    global converage_epoch
    if(converage_epoch < 0.0 ):
        if(logs['acc'] > 0.85):
            converage_epoch = epoch
    if (epoch+1) % (epochs // 5) != 0:
        return
    elif(epoch <= epochs * 3 / 5):
        return
    ##
    start_index = random.randint(0, len(x_train) - 1)
    x_pred = np.array([x_train[start_index]])
    result = copy.deepcopy(x_pred)

    for i in range(generate_length):
        y_pred = model.predict(x_pred, verbose=0)
        # print("y_pred shape:", y_pred.shape)
        result = np.append(result, [y_pred], axis=1)
        # print("before x_pred shape:", x_pred[:,1:maxlen,:,:].shape)
        x_pred = np.append(x_pred[:, 1:maxlen, :, :], [y_pred], axis=1)

    # print('result shape:',result.shape)
    result = np.array(result, dtype=np.bool_)
    # print('result:',result)

    need_length = (generate_length + maxlen) // (96 * 4) * (96 * 4)
    result = result[0]
    result = result[0:need_length]

    # now is stard piano roll
    # print('result shape:', result.shape)
    result = result.reshape((-1, 4, 96, 84, 5))
    # print('result final shape:', result.shape)

    midi_file_name = 'test_train_%d.mid' % (epoch + 1)
    midi_path = os.path.join(midi_log_dir,midi_file_name)
    save_midi(midi_path, result, config)

# In[14]:

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)
# 参照下面代码加一下TensorBoard
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

tb_callbacks = LRTensorBoard(log_dir = TB_log_dir)

print_fn("*"*20+exp_name+"*"*20)
print_fn('x_train shape:'+str(np.shape(x_train)) )
print_fn('y_train shape:'+str(np.shape(y_train)) )

history_callback = model.fit(x_train, y_train,
                             validation_data=(x_val, y_val),
                             verbose=1,
                             batch_size=batch_size,
                             epochs=epochs,
                             callbacks=[tb_callbacks, lr_scheduler, print_callback])

##Predict and get the final result
start_time = time.time()
y_pred = model.predict(x_train, batch_size=batch_size)
speed = (time.time() - start_time ) / ( x_train.shape[0] / batch_size )

y_trues = y_train

final_perplexity = history_callback.history["perplexity"][epochs-1]
final_accuracy = history_callback.history["acc"][epochs-1]
final_cross_entropy = history_callback.history["loss"][epochs-1]
model_size = model.count_params()


print("%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("exp_name", 'model_size', 'final_accuracy', 'final_cross_entropy', 'speed',
                                  'converage_epoch', 'perplexity'))
print(exp_name, model_size, final_accuracy, final_cross_entropy, speed, converage_epoch,final_perplexity)
max_acc_log_line = "%s\t%d\t%f\t%f\t%f\t%d\t%f" % (exp_name, model_size, final_accuracy, final_cross_entropy, speed,
                                                   converage_epoch, final_perplexity)

print(max_acc_log_line, file=open(max_acc_log_path, 'a'))
