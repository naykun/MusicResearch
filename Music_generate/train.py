import keras
from keras.models import load_model,Model
from keras.layers import Dense,Activation,Dropout,Input,LSTM,Reshape,Lambda,RepeatVector,TimeDistributed

from keras.initializers import glorot_uniform
from keras.utils import to_categorical

from keras.optimizers import Adam
from keras import backend as K

import numpy as np

import tensorflow as tf

# import melody_rnn_create_dataset
from sequence_example_lib import *


# def noteseq2seqexmp(noteseqfile):
#     """
#         TODO：解耦config、读取notesequencefile
#         文件：create_dataset
#     """
#     pipeline_inst = melody_rnn_create_dataset.get_pipeline(self.config,
#                                                            eval_ratio=0.0)
#     result = pipeline_inst.transform(note_sequence)

def seqexmp2inputs(seqexmp_file_paths,mode,batch_size, input_size):
    inputs, labels, lengths = get_padded_batch(
          seqexmp_file_paths, batch_size, input_size,
          shuffle=mode == 'train')
    return inputs,labels,lengths





# number of hidden states
n_a = 64
# number of unique values
n_values = 38


def getmodel(input,lengths,n_a,n_values):
    
    # reshapor = Reshape((1,n_values))
    # LSTM_cell = LSTM(n_a,return_state = True,)
    LSTM_cell = LSTM(n_a,return_sequences = True)
    # densor = Dense(n_values,activation='softmax')
    tddensor = TimeDistributed(Dense(n_values, activation='softmax'))

    # X = Input(shape=(Tx,n_values))
    X = Input(tensor =input)

    a0 = Input(shape=(n_a,),name='a0')
    c0 = Input(shape=(n_a,),name='c0')

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

batch_size = 32
total_num = 1600

def train():
    sess = K.get_session()
    seqexmp_file_paths = ['/media/naykun/C836A2D236A2C134/TsinghuaResearch/Wikifonia_basic_rnn_sequence_examples/training_melodies.tfrecord' ]
    # import ipdb; ipdb.set_trace()
    inputs,labels,lengths = seqexmp2inputs(seqexmp_file_paths,'train',batch_size, n_values)

    model = getmodel(inputs,lengths, n_a = 64, n_values = n_values)
    # labels = tf.cast(labels, tf.int32)
    labels = tf.one_hot(labels, n_values)
    # labels_flat = flatten_maybe_padded_sequences(labels, lengths)
    opt = Adam(lr = 0.01, beta_1 = 0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer = opt,loss='categorical_crossentropy',metrics=['accuracy'],target_tensors=[labels])
    model.summary()
    # m = 60

    # a0 = np.zeros((m,n_a))
    # c0 = np.zeros((m,n_a))

    # to be modified
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    model.fit(epochs=100,steps_per_epoch=int(np.ceil( total_num/ float(batch_size))),)
    

if __name__ == '__main__':
    train()