import keras
from keras.models import load_model,Model
from keras.layers import Dense,Activation,Dropout,Input,LSTM,Reshape,Lambda,RepeatVector,TimeDistributed

from keras.initializers import glorot_uniform
from keras.utils import to_categorical

from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import numpy as np

import tensorflow as tf

from sequence_example_lib import *

import os


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('layer_size', 64,
                            'The number of hidden states.')
tf.app.flags.DEFINE_integer('notes_range', 38,
                            'The length of the range of notes.')
tf.app.flags.DEFINE_integer('batch_size', 32,
                            'Batch size.')
tf.app.flags.DEFINE_integer('epochs', 100,
                            'Epochs.')
tf.app.flags.DEFINE_integer('window_size', 0,
                            'Embedding Length.')
tf.app.flags.DEFINE_string('sequence_example_dir', '',
                           'The directory of sequence example for training.')                      
tf.app.flags.DEFINE_boolean('eval', False,
                           'Evaluate the model.')  

def check():
    if(FLAGS.sequence_example_dir == ''):
        print("ERROR: --sequence_example_dir required.")
        return False
    return True

def get_train_model(input, lengths, layer_size, notes_range):
    LSTM_cell = LSTM(layer_size, return_sequences = True)
    tddensor = TimeDistributed(Dense(notes_range, activation='softmax'))
    X = Input(tensor =input)

    a0 = Input(shape=(layer_size,),name='a0') #LSTM acitvation value
    c0 = Input(shape=(layer_size,),name='c0') #LSTM cell value

    a = a0
    c = c0

    lstm = LSTM_cell(X)
    outputs = tddensor(lstm)
    
    '''
    for t in range(Tx):
         x = Lambda(lambda x: x[:,t,:])(X)
         x = reshapor(x)
         a, _, c = LSTM_cell(x,initial_state=[a,c])
         out = densor(a)
         outputs.append(out)
    logits_flat = flatten_maybe_padded_sequences(outputs, lengths
    import ipdb; ipdb.set_trace()
    '''
    
    model = Model(inputs=[X],outputs=outputs)
    return model


def lr_schedule(epoch):
    #Learning Rate Schedule
    lr = 1e-1
    epochs = FLAGS.epochs
    if epoch >= epochs * 0.9:
        lr *= 0.5e-3
    elif epoch >= epochs * 0.8:
        lr *= 1e-3
    elif epoch >= epochs * 0.6:
        lr *= 1e-2
    elif epoch >= epochs * 0.4:
        lr *= 1e-1
    print('Learning rate: ', lr)

    lr = 1e-3
    return lr

def train():
    # Load parameters
    layer_size = FLAGS.layer_size
    batch_size = FLAGS.batch_size
    notes_range = FLAGS.notes_range
    epochs = FLAGS.epochs
    sequence_example_file_paths = [FLAGS.sequence_example_dir]

    sess = K.get_session()
  
    inputs, labels, lengths = get_padded_batch(sequence_example_file_paths, batch_size, notes_range,shuffle = True )

    model = get_train_model(inputs,lengths, layer_size = layer_size, notes_range = notes_range)
    labels = tf.one_hot(labels, notes_range)

    optimizer = RMSprop(lr=lr_schedule(0))
    #optimizer = Adam(lr = 0.01, beta_1 = 0.9, beta_2=0.999, decay=0.01)
    model.compile(  optimizer = optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    target_tensors=[labels])
    model.summary()
    # m = 60
    # a0 = np.zeros((m,layer_size))
    # c0 = np.zeros((m,layer_size))
    # to be modified
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    dataset_size = count_records(sequence_example_file_paths)
    
    exp_name = 'LayerSize%d_BatchSize%d_Epochs%d' % (layer_size, batch_size, epochs)
    model_log_dir = os.path.join('logdir',exp_name)
    

    tb_log_dir = os.path.join('TB_logdir',exp_name)
    from keras.callbacks import TensorBoard
    tb_callbacks = TensorBoard(log_dir = tb_log_dir)

    lr_scheduler = LearningRateScheduler(lr_schedule)
    history_callback = model.fit(  
                epochs=epochs,
                callbacks=[tb_callbacks, lr_scheduler],
                steps_per_epoch=int(np.ceil( dataset_size/ float(batch_size))),)


    acc_history = history_callback.history["acc"]
    max_acc = str(np.max(acc_history))
    max_acc_log_dir = os.path.join('Max_Acc_logdir','Max_Acc.txt')
    try:
        os.makedirs('Max_Acc_logdir')
    except:
        pass
    print('Max accuracy:',max_acc)
    print(exp_name + '\t' + str(layer_size) + '\t' + str(batch_size) 
            + '\t' + str(epochs) + '\t' + max_acc, file = open(max_acc_log_dir, 'a'))

    try:
        os.makedirs(model_log_dir)
    except:
        pass
    model_weights_dir = os.path.join(model_log_dir, exp_name+'_weights.h5')
    model.save_weights(model_weights_dir)

    # Clean up the TF session.
    coord.request_stop()
    coord.join(threads)
    K.clear_session()


if __name__ == '__main__':
    if(check()):
        if(FLAGS.eval == True):
            evaluate()
        else:
            train()


def evaluate():
    # Load parameters
    layer_size = FLAGS.layer_size
    batch_size = FLAGS.batch_size
    notes_range = FLAGS.notes_range
    epochs = FLAGS.epochs
    sequence_example_file_paths = [FLAGS.sequence_example_dir]

    exp_name = 'LayerSize%d_BatchSize%d_Epochs%d' % (layer_size, batch_size, epochs)
    model_log_dir = os.path.join('logdir',exp_name)
    model_weights_dir = os.path.join(model_log_dir, exp_name+'_weights.h5')

    sess = K.get_session()
    
    inputs, labels, lengths = get_padded_batch(sequence_example_file_paths, batch_size, notes_range,shuffle = True )

    model = get_train_model(inputs,lengths, layer_size = layer_size, notes_range = notes_range)
    labels = tf.one_hot(labels, notes_range)
    optimizer = RMSprop(lr=lr_schedule(0))
    model.compile(  optimizer = optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    target_tensors=[labels])

    model.load_weights(model_weights_dir)
    model.trainable = False
    model.summary()

    # m = 60
    # a0 = np.zeros((m,layer_size))
    # c0 = np.zeros((m,layer_size))
    # to be modified
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    dataset_size = count_records(sequence_example_file_paths)

    exp_name = 'LayerSize%d_BatchSize%d_Epochs%d' % (layer_size, batch_size, epochs)
    model_log_dir = os.path.join('logdir',exp_name)
    
    tb_log_dir = os.path.join('TB_logdir',exp_name)

    from keras.callbacks import TensorBoard
    tb_callbacks = TensorBoard(log_dir = tb_log_dir)

    history_callback = model.fit(  
                epochs=epochs,
                callbacks=[tb_callbacks],
                steps_per_epoch=int(np.ceil( dataset_size/ float(batch_size))),)

    acc_history = history_callback.history["acc"]

    max_acc = str(np.max(acc_history))
    print('Max accuracy:',max_acc)
    print(exp_name + '\t' + str(layer_size) + '\t' + str(batch_size) 
            + '\t' + str(epochs) + '\t' + max_acc)

    # Clean up the TF session.
    coord.request_stop()
    coord.join(threads)
    K.clear_session()
