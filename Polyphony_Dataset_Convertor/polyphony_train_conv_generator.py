from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical


import time
import numpy as np
import random
import sys
import os
import keras

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import LambdaCallback

from my_to_midi import *

import pickle as pkl

from polyphony_dataset_convertor import *


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 1024, 'LSTM Layer Units Number')
tf.app.flags.DEFINE_integer('epochs', 150, 'Total epochs')
tf.app.flags.DEFINE_integer('maxlen', 64, 'Max length of a sentence')
tf.app.flags.DEFINE_integer('generate_length', 3200, 'Number of steps of generated music')
tf.app.flags.DEFINE_integer('units', 512, 'LSTM Layer Units Number')
tf.app.flags.DEFINE_integer('dense_size', 0, 'Dense Layer Size')
tf.app.flags.DEFINE_integer('step', 1, 'Step length when building dataset')
tf.app.flags.DEFINE_integer('embedding_length', 1, 'Embedding length')
tf.app.flags.DEFINE_string('dataset_name', 'Bach', 'Dataset name will be the prefix of exp_name')
tf.app.flags.DEFINE_string('dataset_dir', '/home/ouyangzhihao/sss/Mag/Mag_Data/Poly/Poly_List_Datasets/', 'Dataset Directory, which should contain name_train.txt and name_eval.txt')

# In[2]:~/sss/Mag/Mag_Data


batch_size = FLAGS.batch_size
epochs = FLAGS.epochs
units = FLAGS.units
dense_size = FLAGS.dense_size


maxlen = FLAGS.maxlen
generate_length = FLAGS.generate_length
step = FLAGS.step
embedding_length = FLAGS.embedding_length
dataset_name = FLAGS.dataset_name
dataset_dir = FLAGS.dataset_dir


date_and_time = time.strftime('%Y-%m-%d_%H%M%S')

exp_name = "A_3Layer_%s_batchS%d_epochs%d_maxL%d_step%d_embeddingL%d_%s" % (dataset_name,
                                                                        batch_size, epochs, maxlen, step,
                                                                        embedding_length, date_and_time)
from ConvModel import *
from ConvResModel import *
from ConvOtherStructureModel import *
from keras import backend as K


# model = get_conv1d_model_naive(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_conv1d_model(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_resnet_model_naive(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_resNet_model(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_lstm_model(input_shape=train_input_shape,output_shape = train_output_shape)
# model = resnet_v1_110(input_shape=train_input_shape,output_shape = train_output_shape)
# get_model_fn = get_conv1d_model_b_small

# get_model_fn = get_conv1d_model_naive_big
get_model_fn = get_conv1d_model_a
# get_model_fn = get_conv1d_model_naive

# get_model_fn = get_conv1d_model

vector_dim = 259
# train_dataset_path = os.path.join(dataset_dir, dataset_name+'_train.pkl')
# eval_dataset_path = os.path.join(dataset_dir, dataset_name+'_eval.pkl')
train_dataset_path = '/home/ouyangzhihao/sss/Mag/Mag_Data/Poly/Poly_List_Datasets/Bach_new_train.pkl'
eval_dataset_path = '/home/ouyangzhihao/sss/Mag/Mag_Data/Poly/Poly_List_Datasets/Bach_new_eval.pkl'

with open(train_dataset_path, "rb") as train_file:
    train_data = pkl.load(train_file)
    '''
    temp = []
    for i in train_data:
        temp = temp + i[1:len(i)-1]
    '''
    train_data = np.array(train_data)
    train_file.close()

print('Train dataset shape:', train_data.shape)

with open(eval_dataset_path, "rb") as eval_file:
    eval_data = pkl.load(eval_file)
    '''
    temp = []
    for i in eval_data:
        temp = temp + i[1:len(i)-1]
    '''
    eval_data = np.array(eval_data)
    eval_file.close()

print('Eval dataset shape:', eval_data.shape)


# import ipdb; ipdb.set_trace()
# print(train_data[0].shape)
# print(train_data[0])


# train_data = train_data[0:10000]
# eval_data = eval_data[0:2000]

# In[6]:
log_root = '/unsullied/sharefs/ouyangzhihao/Share/LSTM/Text_Generation_Capacity/Code/Music_Research_Exp/Music_Text_Gneration/9_16/multi_track_bach'
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


# In[7]:

train_data = train_data[:int(len(train_data)/1)]
eval_data = eval_data[:int(len(eval_data)/1)]

# In[8]:


def print_fn(str):
    print(str)
    console_log_file = os.path.join(console_log_dir, 'console_output.txt')
    with open(console_log_file, 'a+') as f:
        print(str, file=f)

def lr_schedule(epoch):
    # Learning Rate Schedule

    lr = 1e-3
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
train_input_shape = ( maxlen, vector_dim )

model = get_model_fn(input_shape=train_input_shape,output_shape = train_output_shape)

def perplexity(y_trues, y_preds):
    cross_entropy = K.categorical_crossentropy(y_trues, y_preds)
    perplexity = K.pow(2.0, cross_entropy)

    #Another Method
    # oneoverlog2 = 1.442695
    # result = K.log(x) * oneoverlog2
    return perplexity

optimizer = Adam(lr=lr_schedule(0))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy',perplexity])
model.summary(print_fn=print_fn)

# In[11]:


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[12]:


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

def save_model(epoch):
    model_name = "epoch%d.h5" % (epoch)
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

    # Function invoked at end of each epoch. Prints generated data.
    if(epoch+1 == epochs):
        save_model(epochs)
        # for i in range(10):
        #     start_index = random.randint(0, len(train_data) - maxlen - 1)
        #     generate_more_midi(i,train_data,diversity=0.2,start_index=start_index)
        # for i in range(10):
        #     start_index = random.randint(0, len(eval_data) - maxlen - 1)
        #     generate_more_midi(i, eval_data, diversity=0.2, start_index=start_index, eval_input = True)
    return
    global converage_epoch
    if(converage_epoch < 0.0 ):
        if(logs['acc'] > 0.85):
            converage_epoch = epoch
    if (epoch+1) % (epochs // 5) != 0:
        return
    elif(epoch <= epochs * 3 / 5):
        return

    print_fn("")
    print_fn('----- Generating Music after Epoch: %d' % epoch)

    start_index = random.randint(0, len(train_data) - maxlen - 1)

    # baseline_music(epoch=epoch, data=eval_data, start_index=0)
    # baseline_music(epoch=epoch, data=eval_data, start_index=start_index)
    # baseline_music(epoch=epoch, data=train_data, start_index=start_index, is_train=True)

    for diversity in [0.2, 1.2]:
        # generate_music(epoch=epoch, data=eval_data, diversity=diversity, start_index=0)
        # generate_music(epoch=epoch, data=eval_data, diversity=diversity, start_index=start_index)
        generate_music(epoch=epoch, data=train_data, diversity=diversity, start_index=start_index, is_train=True)


##Data generator
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import keras
# data = np.array([[i] for i in range(50)])
# targets = np.array([[i] for i in range(50)])

train_data = keras.utils.to_categorical(train_data, num_classes=vector_dim)
eval_data = keras.utils.to_categorical(eval_data, num_classes=vector_dim)

train_generator = TimeseriesGenerator(train_data[:len(train_data)-1], train_data[1:],
                               length=maxlen, sampling_rate=1,
                               batch_size=batch_size)
eval_generator = TimeseriesGenerator(eval_data[:len(eval_data)-1], eval_data[1:],
                               length=maxlen, sampling_rate=1,
                               batch_size=batch_size)

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

history_callback = model.fit_generator(train_generator,workers=4,
                             validation_data=eval_generator,
                             verbose=1,
                             epochs=epochs,
                             steps_per_epoch=epochs/batch_size,
                             callbacks=[tb_callbacks, lr_scheduler, print_callback],
                                       )

##Predict and get the final result
start_time = time.time()
y_pred = model.predict_generator(train_generator)

speed = (time.time() - start_time ) / ( train_data.shape[0] * maxlen / batch_size )

# print('Finished')
# print(model.count_params())
# print(y_pred.shape)
# print(history_callback)
# import ipdb; ipdb.set_trace()
# history_callback.history.items()

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


'''


rlaunch --cpu=4 --gpu=1 --memory=16000 --preemptible=no bash


python3 polyphony_train.py --batch_size=512 \
    --epochs=20 \
    --units=512 \
    --maxlen=16 \
    --generate_length=400 \
    --dense_size=0 \
    --step=1 \
    --embedding_length=1 \
    --dataset_name=Bach_new \
    --dataset_dir=datasets/Bach/




rlanuch --cpu=4 -- python3 --

'''