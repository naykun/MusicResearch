#coding:utf-8
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
tf.app.flags.DEFINE_integer('Ty', 100,
                            'The number of hidden states.')
tf.app.flags.DEFINE_integer('epochs', 10,
                            'Epochs.')
tf.app.flags.DEFINE_integer('embedding_len', 1,
                            'Embedding Length.')
tf.app.flags.DEFINE_string('sequence_example_train_dir', 'D:\\TsinghuaResearch\\Wikifonia_basic_rnn_sequence_examples\\train',
                           'The directory of sequence example for training.')
tf.app.flags.DEFINE_string('sequence_example_val_dir', 'D:\\TsinghuaResearch\\Wikifonia_basic_rnn_sequence_examples\\eval',
                           'The directory of sequence example for validation.')                         
tf.app.flags.DEFINE_integer('maxlen', 511,
                            'max timesteps')



def get_train_model(layer_size,notes_range,embedding_len,maxlen):


    reshapor = Reshape((1,notes_range*embedding_len))
    LSTM_cell = LSTM(layer_size, return_state = True)
    # tddensor = TimeDistributed(Dense(notes_range, activation='softmax'))
    densor = Dense(notes_range, activation='softmax')
    X = Input(shape=(maxlen,notes_range))

    a0 = Input(shape=(layer_size,),name='a0') #LSTM acitvation value
    c0 = Input(shape=(layer_size,),name='c0') #LSTM cell value
    # a0 = np.zeros((m,layer_size))
    # c0 = np.zeros((m,layer_size))

    a = a0
    c = c0

    # lstm = LSTM_cell(X)
    # outputs = tddensor(lstm)
    outputs = []

    for t in range(maxlen-embedding_len):
        x = Lambda(lambda x: x[:,t:t+embedding_len,:])(X)
        x = reshapor(x)
        if t==0:
            a, _, c = LSTM_cell(x,initial_state=[a0,c0])
        else:
            a, _, c = LSTM_cell(x,initial_state=[a,c])
        out = densor(a)
        outputs.append(out)
    # logits_flat = flatten_maybe_padded_sequences(outputs, lengths
    # import ipdb; ipdb.set_trace()


    model = Model(inputs=[X,a0,c0],outputs=outputs)

    return model,LSTM_cell,reshapor,densor


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
    embedding_len = FLAGS.embedding_len
    maxlen = FLAGS.maxlen

    train_sequence_example_file_paths = [FLAGS.sequence_example_train_dir]
    val_sequence_example_file_paths = [FLAGS.sequence_example_val_dir]

    X_train, labels_train, _ = get_numpy_from_tf_sequence_example(input_size=38,
                                    sequence_example_file_paths = train_sequence_example_file_paths,
                                    shuffle = False)
    X_val, labels_val, _ = get_numpy_from_tf_sequence_example(input_size=38,
                                    sequence_example_file_paths = val_sequence_example_file_paths,
                                    shuffle = False)
    model,LSTM_cell,reshapor,densor = get_train_model(layer_size = layer_size, notes_range = notes_range,embedding_len=embedding_len,maxlen=maxlen)
    Y_train = to_categorical(labels_train,notes_range)
    Y_val = to_categorical(labels_val,notes_range)
    print(Y_train.shape)
    # import ipdb; ipdb.set_trace()
    optimizer = Adam(lr=lr_schedule(0), beta_1 = 0.9, beta_2=0.999, decay=0.01)
    #optimizer = Adam(lr = 0.01, beta_1 = 0.9, beta_2=0.999, decay=0.01)
    model.compile(  optimizer = optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model.summary()
    m = batch_size
    a0 = np.zeros((m,layer_size))
    c0 = np.zeros((m,layer_size))
    # to be modified
    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess, coord)

    dataset_size = count_records(train_sequence_example_file_paths)
    
    exp_name = 'LayerSize%d_BatchSize%d_Epochs%d' % (layer_size, batch_size, epochs)
    model_log_dir = os.path.join('logdir',exp_name)
    

    tb_log_dir = os.path.join('TB_logdir',exp_name)
    from keras.callbacks import TensorBoard
    tb_callbacks = TensorBoard(log_dir = tb_log_dir)
    # Y_train.split(maxlen-embedding_len,axis=1)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    history_callback = model.fit([X_train, a0, c0], np.split(Y_train,Y_train.shape[1],axis=1), batch_size=batch_size,  
                epochs=epochs,
                callbacks=[tb_callbacks, lr_scheduler],validation_data=([X_val, a0, c0],np.split(Y_val,Y_val.shape[1],axis=1)),
                steps_per_epoch=int(np.ceil( dataset_size/ float(batch_size))))
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
    # coord.request_stop()
    # coord.join(threads)
    # K.clear_session()
    
    return LSTM_cell,reshapor,densor

def get_inference_model(LSTM_cell, reshapor, densor, notes_range = 38,embedding_len=15, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, umber of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """

    slicer = Lambda(lambda x:x[notes_range:])

    # Define the input of your model with a shape 
    x0 = Input(shape=(embedding_len, notes_range))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = reshapor(x0)

    ### START CODE HERE ###
    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []
    
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Step 2.A: Perform one step of LSTM_cell (≈1 line)
        a, _, c = LSTM_cell(x,initial_state=[a,c])
        
        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)

        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
        outputs.append(out)
        
        # Step 2.D: Select the next value according to "out", and set "x" to be the one-hot representation of the
        #           selected value, which will be passed as the input to LSTM_cell on the next step. We have provided 
        #           the line of code you need to do this. 
        x = slicer(x)
        x = merge([x,out],mode='concat')        
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0,a0,c0],outputs=outputs)
    
    ### END CODE HERE ###
    
    return inference_model


if __name__ == '__main__':
    LSTM_cell,reshapor,densor = train()
    inference_model = get_inference_model(LSTM_cell,reshapor,densor,notes_range=FLAGS.notes_range, embedding_len=FLAGS.embedding_len,Ty=FLAGS.Ty)
    inference_model.predict([X_initial,a0,c0],batch_size=predict_batchsize)
    