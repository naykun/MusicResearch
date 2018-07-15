import keras
from keras.models import load_model,Model
from keras.layers import Dense,Activation,Dropout,Input,LSTM,Reshape,Lambda,RepeatVector,TimeDistributed

from keras.initializers import glorot_uniform
from keras.utils import to_categorical

from keras.optimizers import Adam
from keras import backend as K

import numpy as np

import tensorflow as tf

from sequence_example_lib import *


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('layer_size', 64,
                            'The number of hidden states.')
tf.app.flags.DEFINE_integer('notes_range', 38,
                            'The length of the range of notes.')
tf.app.flags.DEFINE_integer('batch_size', 32,
                            'Batch size.')
tf.app.flags.DEFINE_integer('epochs', 100,
                            'Epochs.')
tf.app.flags.DEFINE_string('sequence_example_dir', '',
                           'The directory of sequence example for training')                      

def check():
    if(FLAGS.sequence_example_dir == ''):
        print("ERROR: --sequence_example_dir required.")
        return False
    return True

def getmodel(input, lengths, layer_size, notes_range):
    LSTM_cell = LSTM(layer_size,return_sequences = True)
    tddensor = TimeDistributed(Dense(notes_range, activation='softmax'))
    X = Input(tensor =input)

    a0 = Input(shape=(layer_size,),name='a0') #LSTM acitvation value
    c0 = Input(shape=(layer_size,),name='c0') #LSTM cell value

    a = a0
    c = c0

    lstm = LSTM_cell(X)
    outputs = tddensor(lstm)
    # for t in range(Tx):
    #     x = Lambda(lambda x: x[:,t,:])(X)
    #     x = reshapor(x)
    #     a, _, c = LSTM_cell(x,initial_state=[a,c])
    #     out = densor(a)
    #     outputs.append(out)
    # logits_flat = flatten_maybe_padded_sequences(outputs, lengths)
    # import ipdb; ipdb.set_trace()
    model = Model(inputs=[X],outputs=outputs)
    return model

def train():
    # Load parameters
    layer_size = FLAGS.layer_size
    batch_size = FLAGS.batch_size
    notes_range = FLAGS.notes_range
    epochs = FLAGS.epochs
    sequence_example_file_paths = [FLAGS.sequence_example_dir]

    sess = K.get_session()
  
    inputs, labels, lengths = get_padded_batch(sequence_example_file_paths, batch_size, notes_range,shuffle = True )

    model = getmodel(inputs,lengths, layer_size = layer_size, notes_range = notes_range)
    labels = tf.one_hot(labels, notes_range)
    opt = Adam(lr = 0.01, beta_1 = 0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer = opt,loss='categorical_crossentropy',metrics=['accuracy'],target_tensors=[labels])
    model.summary()
    # m = 60
    # a0 = np.zeros((m,layer_size))
    # c0 = np.zeros((m,layer_size))
    # to be modified
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    dataset_size = count_records(sequence_example_file_paths)
    model.fit(  epochs=epochs,
                steps_per_epoch=int(np.ceil( dataset_size/ float(batch_size))),)

if __name__ == '__main__':
    if(check()):
        train()
