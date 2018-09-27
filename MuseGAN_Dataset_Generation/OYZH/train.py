import tensorflow as tf
import numpy as np
import os
import random
import copy

import keras
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, Reshape
from keras.optimizers import RMSprop, Adam
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import LambdaCallback

import my_config
from my_midi_io import *



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('maxlen', 96, '')
tf.app.flags.DEFINE_integer('generate_length', 10000, '')
tf.app.flags.DEFINE_integer('step', 24, '')
tf.app.flags.DEFINE_integer('batch_size', 2, '')
tf.app.flags.DEFINE_integer('epochs', 2, '')


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


'''
date_and_time = time.strftime('%Y-%m-%d_%H%M%S')

exp_name = "%s_batchS%d_epochs%d_units%d_denseS%d_maxL%d_step%d_embeddingL%d_%s" % (dataset_name,
                                                                        batch_size, epochs, units, dense_size, maxlen, step,
                                                                        embedding_length, date_and_time)
'''


# feature
reshaped_dataset = dataset.reshape((len(dataset), -1, 84, 5))

feature = []
label = []

for now_song in reshaped_dataset:
    for i in range(0, len(now_song), step):
        if (i + maxlen + 1) < len(now_song):
            feature.append(now_song[i:i+maxlen])
            label.append(now_song[i+maxlen])

# In[8]:


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


# In[10]:


# build model
xx = Input(shape=(maxlen, 84, 5))
xxx = Flatten()(xx)
xxx = Dense(84*5, activation='relu')(xxx)
xxx = Reshape((84, 5))(xxx)
model = Model(xx, xxx)
model.summary()

model.compile(loss='mean_squared_error',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# snapshot

log_root = '/unsullied/sharefs/ouyangzhihao/Share/LSTM/Text_Generation_Capacity/Code/Music_Research_Exp/Music_Text_Gneration/9_19/MuseGan/'
def on_epoch_end(epoch, logs):
    start_index = random.randint(0, len(x_train)-1)
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
    
    need_length = (generate_length + maxlen) // (96*4) * (96*4)
    result = result[0]
    result = result[0:need_length]
    
    # now is stard piano roll
    # print('result shape:', result.shape)
    result = result.reshape((-1, 4, 96, 84, 5))
    # print('result final shape:', result.shape)

    save_midi(log_root + 'test_train_%d.mid' % (epoch+1), result, config)


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)


# train
model.fit(  x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            verbose=1,
            epochs=epochs,
            callbacks=[print_callback])


'''

python3 train.py --maxlen=96 \
    --generate_length=1000 \
    --step=24 \
    --batch_size=2 \
    --epochs=2

'''