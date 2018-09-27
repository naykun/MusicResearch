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
from keras import backend as K
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
from math import *

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import LambdaCallback

from my_to_midi import *
from ConvModel import *
from ConvResModel import *
from ConvOtherStructureModel import *

# In[ ]:


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 512, 'LSTM Layer Units Number')
tf.app.flags.DEFINE_integer('epochs', 150, 'Total epochs')
tf.app.flags.DEFINE_integer('maxlen', 64, 'Max length of a sentence')
tf.app.flags.DEFINE_integer('generate_length', 3200, 'Number of steps of generated music')
tf.app.flags.DEFINE_integer('units', 64, 'LSTM Layer Units Number')
tf.app.flags.DEFINE_integer('dense_size', 0, 'Dense Layer Size')
tf.app.flags.DEFINE_integer('step', 1, 'Step length when building dataset')
tf.app.flags.DEFINE_integer('embedding_length', 1, 'Embedding length')
tf.app.flags.DEFINE_string('dataset_name', 'Wikifonia', 'Dataset name will be the prefix of exp_name')
tf.app.flags.DEFINE_string('dataset_dir', '/home/ouyangzhihao/sss/Mag/Mag_Data/TextMelody/Wikifonia/', 'Dataset Directory, which should contain name_train.txt and name_eval.txt')

# In[2]:


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

if(len(sys.argv)!=1):
    order = int(sys.argv[1])
    if(order == 0):
        name_suffix = 'A'
        get_model_fn = get_conv1d_model_a
    if (order == 1):
        name_suffix = 'B'
        get_model_fn = get_conv1d_model_b
    if (order == 2):
        name_suffix = 'C'
        get_model_fn = get_conv1d_model_c
    if (order == 3):
        name_suffix = 'Naive'
        get_model_fn = get_conv1d_model_naive
    if (order == 4):
        name_suffix = 'NaiveBig'
        get_model_fn = get_conv1d_model_naive_big
    if (order == 5):
        name_suffix = 'resNetNaive'
        get_model_fn = get_resnet_model_naive
    if (order == 6):
        name_suffix = 'resNetLocal'
        get_model_fn = get_resNet_model
    if (order == 7):
        name_suffix = 'LSTM'
        get_model_fn = get_lstm_model


exp_name = name_suffix + "_%s_batchS%d_epochs%d_maxL%d_step%d_embeddingL%d_%s" % (dataset_name,
                                                                        batch_size, epochs, maxlen, step,
                                                                        embedding_length, date_and_time)


# model = get_conv1d_model_naive(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_conv1d_model(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_resnet_model_naive
# model = get_resNet_model
# model = get_lstm_model
# model = resnet_v1_110(input_shape=train_input_shape,output_shape = train_output_shape)
# get_model_fn = resnet_v1_110
# get_model_fn = get_conv1d_model_old

# get_model_fn = get_conv1d_model_b_small

# get_model_fn = get_conv1d_model_naive_big
# get_model_fn = get_conv1d_model_b
# get_model_fn = get_conv1d_model_naive

# In[ ]:


train_dataset_path = os.path.join(dataset_dir, dataset_name+'_train.txt')
eval_dataset_path = os.path.join(dataset_dir, dataset_name+'_eval.txt')

with open(train_dataset_path, "r") as train_file:
    train_text = train_file.read().lower()
    train_file.close()

print('Train dataset length:', len(train_text))

with open(eval_dataset_path, "r") as eval_file:
    eval_text = eval_file.read().lower()
    eval_file.close()

print('Eval dataset length:', len(eval_text))

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

# In[6]:



# 9_16
log_root = '/unsullied/sharefs/ouyangzhihao/Share/LSTM/Text_Generation_Capacity/Code/Music_Research_Exp/Music_Text_Gneration/9_16/single_track_Wiki'
log_dir = os.path.join(log_root, "logdir", exp_name)
TB_log_dir = os.path.join(log_root, 'TB_logdir', exp_name)
console_log_dir = os.path.join(log_root, log_dir, "console")
model_log_dir = os.path.join(log_root, 'Model_logdir', exp_name)
text_log_dir = os.path.join(log_root, log_dir, "text")
midi_log_dir = os.path.join(log_root, log_dir, "midi")
midi_log_dir_more = os.path.join(log_root, log_dir, "midi_more")
max_acc_log_path = os.path.join(log_root, "logdir", "max_acc_log.txt")

def make_log_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


dirs = [log_dir, TB_log_dir, console_log_dir, model_log_dir, text_log_dir, midi_log_dir]
make_log_dirs(dirs)





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
    for i in range(0, len(inputs) - embedding_length, step):
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

# train_text = train_text[:int(len(train_text)/10)]
# eval_text = eval_text[:int(len(eval_text)/10)]

print('Vectorization...')
x_train, y_train = get_embedded_data(train_text, maxlen, embedding_length)
x_eval, y_eval = get_embedded_data(eval_text, maxlen, embedding_length)

# In[8]:


def print_fn(str):
    print(str)
    console_log_file = os.path.join(console_log_dir, 'console_output.txt')
    with open(console_log_file, 'a+') as f:
        print(str, file=f)

def lr_schedule(epoch):
    #Learning Rate Schedule
    lr = 1e-2
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
# In[9]:



# In[10]:


# build the model: a single LSTM
print_fn('Build model...')

train_input_shape = x_train.shape[1:]
train_output_shape = (y_train.shape[1])
print('train_input_shape', train_input_shape)
print('train_output_shape', train_output_shape)
print('x_train.shape', x_train.shape)
print('y_train.shape', y_train.shape)


# model = get_model_fn(input_shape=train_input_shape,output_shape = train_output_shape, local_conv = False)
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
        log_name = "epoch%d_train_diversity%02d" % (epoch + 1, int(diversity * 10))
    else:
        if start_index == 0:
            log_name = "epoch%d_first_diversity%02d" % (epoch + 1, int(diversity * 10))
        else:
            log_name = "epoch%d_random_diversity%02d" % (epoch + 1, int(diversity * 10))

    text_log_path = os.path.join(text_log_dir, log_name + ".txt")
    with open(text_log_path, "w") as text_log_file:
        text_log_file.write(generated + "\n")
        text_log_file.close()

    print_fn("Write %s.txt to %s" % (log_name, text_log_dir))

    events = text_to_events(generated)
    events_to_midi('basic_rnn', events, midi_log_dir, log_name)

    print_fn("Write %s.midi to %s" % (log_name, midi_log_dir))

    model_name = "epoch%d.h5" % (epoch+1)
    model_path = os.path.join(model_log_dir, model_name)
    model.save(model_path)
    print_fn("Save model %s.h5 to %s" % (model_name, model_log_dir))


def baseline_music(epoch, text, start_index, is_train=False):
    print_fn('----- baseline')

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence

    generated += eval_text[start_index + maxlen: min(len(text), start_index + maxlen + generate_length)]
    sys.stdout.write(generated)

    if is_train:
        log_name = "epoch%d_train_baseline" % (epoch + 1)
    else:
        if start_index == 0:
            log_name = "epoch%d_first_baseline" % (epoch + 1)
        else:
            log_name = "epoch%d_random_baseline" % (epoch + 1)

    text_log_path = os.path.join(text_log_dir, log_name + ".txt")
    with open(text_log_path, "w") as text_log_file:
        text_log_file.write(generated + "\n")
        text_log_file.close()

    print_fn("Write %s.txt to %s" % (log_name, text_log_dir))

    events = text_to_events(generated)
    events_to_midi('basic_rnn', events, midi_log_dir, log_name)

    print_fn("Write %s.midi to %s" % (log_name, midi_log_dir))


def generate_more_midi(id, text, diversity, start_index, eval_input = False):
    print_fn('----- diversity: %.1f' % diversity)

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print_fn('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)
    # generate_length = 3200
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
    if(eval_input):
        log_name = "id%d_eval_diver%02d" % (id + 1, int(diversity * 10))
    else:
        log_name = "id%d_train_diversity%02d" % (id + 1, int(diversity * 10))

    events = text_to_events(generated)
    events_to_midi('basic_rnn', events, midi_log_dir_more, log_name)
    print_fn("Write %s.midi to %s" % (log_name, midi_log_dir_more))


converage_epoch = -1
def on_epoch_end(epoch, logs):
    print('OYZH epoch',epoch)

    if(epoch+1 == epochs):
        for i in range(10):
            start_index = random.randint(0, len(train_text) - maxlen - 1)
            generate_more_midi(i,train_text,diversity=0.2,start_index=start_index)
        for i in range(10):
            start_index = random.randint(0, len(eval_text) - maxlen - 1)
            generate_more_midi(i, eval_text, diversity=0.2, start_index=start_index, eval_input = True)

    # return
    # Function invoked at end of each epoch. Prints generated data.
    global converage_epoch
    if(converage_epoch < 0.0 ):
        if(logs['acc'] > 0.85):
            converage_epoch = epoch
    # Function invoked at end of each epoch. Prints generated text.
    if (epoch+1) % (epochs // 5) != 0:
        return
    elif(epoch <= epochs * 4 / 5):
        return

    print_fn("")
    print_fn('----- Generating Music after Epoch: %d' % epoch)

    start_index = random.randint(0, len(eval_text) - maxlen - 1)

    baseline_music(epoch=epoch, text=eval_text, start_index=0)
    baseline_music(epoch=epoch, text=eval_text, start_index=start_index)
    baseline_music(epoch=epoch, text=eval_text, start_index=start_index, is_train=True)

    for diversity in [0.2, 1.2]:
        generate_music(epoch=epoch, text=eval_text, diversity=diversity, start_index=0)
        generate_music(epoch=epoch, text=eval_text, diversity=diversity, start_index=start_index)
        generate_music(epoch=epoch, text=eval_text, diversity=diversity, start_index=start_index, is_train=True)




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
                             validation_data=(x_eval, y_eval),
                             verbose=1,
                             batch_size=batch_size,
                             epochs=epochs,
                             callbacks=[tb_callbacks, lr_scheduler, print_callback])

##Predict and get the final result
start_time = time.time()
y_pred = model.predict(x_train, batch_size=batch_size)
speed = (time.time() - start_time ) / ( x_train.shape[0] / batch_size )

y_trues = y_train

final_perplexity = get_perplexity(y_trues, y_pred)
final_cross_entropy = get_cross_entropy(y_trues, y_pred)
final_accuracy = get_accuracy(y_trues, y_pred)
model_size = model.count_params()


# acc_history = history_callback.history["acc"]
# max_acc = np.max(acc_history)
# print_fn('Experiment %s max accuracy:%f' % (exp_name, max_acc))
# max_acc_log_line = "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%f" % (exp_name,
#                                                    epochs, units, dense_size, maxlen, step, embedding_length, max_acc)


print("%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("exp_name", 'model_size', 'final_accuracy', 'final_cross_entropy', 'speed',
                                  'converage_epoch', 'perplexity'))
print(exp_name, model_size, final_accuracy, final_cross_entropy, speed, converage_epoch,final_perplexity)
max_acc_log_line = "%s\t%d\t%f\t%f\t%f\t%d\t%f" % (exp_name, model_size, final_accuracy, final_cross_entropy, speed,
                                                   converage_epoch, final_perplexity)

print(max_acc_log_line, file=open(max_acc_log_path, 'a'))



'''

python3 music_text_generator.py --batch_size=1024 \
    --epochs=40 \
    --units=512 \
    --maxlen=16 \
    --generate_length=400 \
    --dense_size=0 \
    --step=1 \
    --embedding_length=16 \
    --dataset_name=Bach \
    --dataset_dir=datasets/Bach/




rlanuch --cpu=4 -- python3 --

'''