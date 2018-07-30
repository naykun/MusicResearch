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


import time
import numpy as np
import random
import sys
import os
from math import *

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import LambdaCallback

from my_to_midi import *

# In[ ]:


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 1024, 'LSTM Layer Units Number')
tf.app.flags.DEFINE_integer('epochs', 5, 'Total epochs')
tf.app.flags.DEFINE_integer('maxlen', 48, 'Max length of a sentence')
tf.app.flags.DEFINE_integer('generate_length', 400, 'Number of steps of generated music')
tf.app.flags.DEFINE_integer('units', 128, 'LSTM Layer Units Number')
tf.app.flags.DEFINE_integer('dense_size', 0, 'Dense Layer Size')
tf.app.flags.DEFINE_integer('step', 8, 'Step length when building dataset')
tf.app.flags.DEFINE_integer('embedding_length', 1, 'Embedding length')
tf.app.flags.DEFINE_string('dataset_name', 'Bach', 'Dataset name will be the prefix of exp_name')
tf.app.flags.DEFINE_string('dataset_dir', 'datasets/Bach/', 'Dataset Directory, which should contain name_train.txt and name_eval.txt')

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

exp_name = "%s_batchS%d_epochs%d_units%d_denseS%d_maxL%d_step%d_embeddingL%d_%s" % (dataset_name,
                                                                        batch_size, epochs, units, dense_size, maxlen, step,
                                                                        embedding_length, date_and_time)

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


log_dir = os.path.join("logdir/", exp_name)
TB_log_dir = os.path.join('TB_logdir/', exp_name)
console_log_dir = os.path.join(log_dir, "console")
model_log_dir = os.path.join('Model_logdir', exp_name)
text_log_dir = os.path.join(log_dir, "text")
midi_log_dir = os.path.join(log_dir, "midi")


def make_log_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


dirs = [log_dir, TB_log_dir, console_log_dir, model_log_dir, text_log_dir, midi_log_dir]
make_log_dirs(dirs)

max_acc_log_path = os.path.join("logdir/", "max_acc_log.txt")



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
    # Learning Rate Schedule

    lr = 1e-1
    if epoch >= epochs * 0.9:
        lr *= 0.5e-3
    elif epoch >= epochs * 0.8:
        lr *= 1e-3
    elif epoch >= epochs * 0.6:
        lr *= 1e-2
    elif epoch >= epochs * 0.4:
        lr *= 1e-1
    print_fn('Learning rate: ', lr)

    lr = 1e-3
    return lr


# In[9]:



# In[10]:


# build the model: a single LSTM
print_fn('Build model...')

model = Sequential()

if dense_size != 0:
    model.add(Dense(dense_size,input_shape=(maxlen, len(chars)*embedding_length )))
    model.add(LSTM(units))
else:
    model.add(LSTM(units, input_shape=(maxlen, len(chars)*embedding_length )))

model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = Adam(lr=lr_schedule(0))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
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


def generate_music(epoch, text, diversity, start_index):
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


def baseline_music(epoch, text, start_index):
    print_fn('----- baseline')

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence

    generated += eval_text[start_index + maxlen: min(len(text), start_index + maxlen + generate_length)]
    sys.stdout.write(generated)

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


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    if (epoch+1) % (epochs // 5) != 0:
        return

    print_fn("")
    print_fn('----- Generating Music after Epoch: %d' % epoch)

    start_index = random.randint(0, len(eval_text) - maxlen - 1)

    baseline_music(epoch=epoch, text=eval_text, start_index=0)
    baseline_music(epoch=epoch, text=eval_text, start_index=start_index)
    for diversity in [0.2, 0.5, 0.8, 1.0, 1.2]:
        generate_music(epoch=epoch, text=eval_text, diversity=diversity, start_index=0)
        generate_music(epoch=epoch, text=eval_text, diversity=diversity, start_index=start_index)


# In[14]:

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)
# 参照下面代码加一下TensorBoard
tb_callbacks = TensorBoard(log_dir=TB_log_dir)

print_fn("*"*20+exp_name+"*"*20)
print_fn('x_train shape:'+str(np.shape(x_train)) )
print_fn('y_train shape:'+str(np.shape(y_train)) )

history_callback = model.fit(x_train, y_train,
                             validation_data=(x_eval, y_eval),
                             verbose=1,
                             batch_size=batch_size,
                             epochs=epochs,
                             callbacks=[tb_callbacks, lr_scheduler, print_callback])

acc_history = history_callback.history["acc"]
max_acc = np.max(acc_history)
print_fn('Experiment %s max accuracy:%f' % (exp_name, max_acc))
max_acc_log_line = "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%f" % (exp_name,
                                                   epochs, units, dense_size, maxlen, step, embedding_length, max_acc)

print(max_acc_log_line, file=open(max_acc_log_path, 'a'))


'''

python3 music_text_generator.py --batch_size=1024 \
    --epochs=10 \
    --units=128 \
    --maxlen=48 \
    --generate_length=400 \
    --dense_size=3 \
    --step=8 \
    --embedding_length=4 \
    --dataset_name=Bach \
    --dataset_dir=datasets/Bach/




rlanuch --cpu=4 -- python3 --

'''