'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from ConvModel import *
import numpy as np
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28
maxlen = img_rows
exp_name = 'naive_conv1d_model_WinSize%d_BS%d_epoch%d' % (maxlen, batch_size, epochs)
max_acc_log = '../res/mnist_acc.txt'

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

train_input_shape = (x_train.shape[1],x_train.shape[2])
train_output_shape = (y_train.shape[1])

# model = get_conv1d_model(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_conv1d_model_old(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_two_pipeline_model(input_shape=train_input_shape,output_shape = train_output_shape)

model = get_naive_conv1d_model(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_conv1d_model_simple(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_resNet_model(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_resNet_model(input_shape=train_input_shape, output_shape = train_output_shape)


# model = get_lstm_model(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_conv1d_resnet(input_shape=train_input_shape,output_shape = train_output_shape)
# model = get_complex_model(input_shape_melody=train_input_shape, input_shape_accom=train_input_shape,  output_shape=train_output_shape)
# model = get_complex_model_resNet_melody(input_shape_melody=train_input_shape, input_shape_accom=train_input_shape,  output_shape=train_output_shape)

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history_callback = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)


acc_history = history_callback.history["acc"]

max_acc = str(np.max(acc_history))
print('Train Exp max accuracy:', max_acc)

print('-------\n' + exp_name  + '\t' + str(maxlen) + '\ttrain max_acc' + max_acc , file=open(max_acc_log, 'a'))

print('Test loss:', score[0], file=open(max_acc_log, 'a'))
print('Test accuracy:', score[1], file=open(max_acc_log, 'a'))

