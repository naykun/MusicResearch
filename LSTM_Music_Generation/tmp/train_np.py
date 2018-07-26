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
tf.app.flags.DEFINE_integer('epochs', 1,
                            'Epochs.')
tf.app.flags.DEFINE_integer('embedding_len', 1,
                            'Embedding Length.')
tf.app.flags.DEFINE_string('sequence_example_train_file', 'Wikifonia_basic_rnn_sequence_examples/training_melodies.tfrecord',
                           'The directory of sequence example for training.')
tf.app.flags.DEFINE_string('sequence_example_eval_file', 'Wikifonia_basic_rnn_sequence_examples/eval_melodies.tfrecord',
                           'The directory of sequence example for validation.')                         
tf.app.flags.DEFINE_integer('maxlen', 20,
                            'max timesteps')


def get_train_model(FLAGS):
    
    layer_size = FLAGS.layer_size
    notes_range = FLAGS.notes_range
    embedding_len = FLAGS.embedding_len
    maxlen = FLAGS.maxlen

    reshapor = Reshape((1,notes_range*embedding_len))
    LSTM_cell = LSTM(layer_size, return_state = True)
    # tddensor = TimeDistributed(Dense(notes_range, activation='softmax'))
    densor = Dense(notes_range, activation='softmax')
    X = Input(shape=(maxlen-embedding_len,notes_range*embedding_len))

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
        x = Lambda(lambda x: x[:,t,:])(X)
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


total_epochs = 0

def lr_schedule(epoch):
    #Learning Rate Schedule
    lr = 1e-1
    global total_epochs
    epochs = total_epochs
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
def dataset_embedding(X,labels,maxlen,embedding_len):
    X_embedded = []
    for i in range(maxlen - embedding_len ):
        X_embedded.append(np.reshape(X[:,i:i+embedding_len,:],(X.shape[0],1,-1)))

    X_embedded = np.concatenate(X_embedded,axis=1)
    label_embedded = labels[:,embedding_len:maxlen]
    return X_embedded,label_embedded


def train(FLAGS):
    # Load parameters
    layer_size = FLAGS.layer_size
    batch_size = FLAGS.batch_size
    notes_range = FLAGS.notes_range

    global total_epochs
    epochs = total_epochs = FLAGS.epochs
    
    embedding_len = FLAGS.embedding_len
    maxlen = FLAGS.maxlen

    train_sequence_example_file_paths = [FLAGS.sequence_example_train_file]
    val_sequence_example_file_paths = [FLAGS.sequence_example_eval_file]

    X_train, labels_train, _ = get_numpy_from_tf_sequence_example(input_size=38,
                                    sequence_example_file_paths = train_sequence_example_file_paths,
                                    shuffle = False)
    X_val, labels_val, _ = get_numpy_from_tf_sequence_example(input_size=38,
                                    sequence_example_file_paths = val_sequence_example_file_paths,
                                    shuffle = False)
    X_train,labels_train = dataset_embedding(X_train, labels_train, maxlen, embedding_len)
    X_val,labels_val = dataset_embedding(X_val, labels_val, maxlen, embedding_len)


    model,LSTM_cell,reshapor,densor = get_train_model(FLAGS)
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
    # to be modified
    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess, coord)

    train_dataset_size = count_records(train_sequence_example_file_paths)
    
    m_train = train_dataset_size
    a0_train = np.zeros((m_train,layer_size))
    c0_train = np.zeros((m_train,layer_size))
    
    val_dataset_size = count_records(val_sequence_example_file_paths)
    
    m_val = val_dataset_size
    a0_val = np.zeros((m_val,layer_size))
    c0_val = np.zeros((m_val,layer_size))

    exp_name = 'LayerSize%d_BatchSize%d_Epochs%d' % (layer_size, batch_size, epochs)
    model_log_file = os.path.join('logdir',exp_name)
    

    tb_log_file = os.path.join('TB_logdir',exp_name)
    from keras.callbacks import TensorBoard
    tb_callbacks = TensorBoard(log_dir = tb_log_file)
    # Y_train.split(maxlen-embedding_len,axis=1)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    Y_train = [np.squeeze(x) for x in np.split(Y_train,Y_train.shape[1],axis=1)]
    Y_val = [np.squeeze(x) for x in np.split(Y_val,Y_val.shape[1],axis=1)]
    history_callback = model.fit([X_train, a0_train, c0_train], Y_train,batch_size=batch_size,
                epochs=epochs,
                callbacks=[tb_callbacks, lr_scheduler],validation_data=([X_val, a0_val, c0_val],Y_val))
    for key in history_callback.history:
        print(key)
    # acc_history = history_callback.history["acc"]
    # max_acc = str(np.max(acc_history))
    # max_acc_log_file = os.path.join('Max_Acc_logdir','Max_Acc.txt')
    # try:
    #     os.makedirs('Max_Acc_logdir')
    # except:
    #     pass
    # print('Max accuracy:',max_acc)
    # print(exp_name + '\t' + str(layer_size) + '\t' + str(batch_size)
    #         + '\t' + str(epochs) + '\t' + max_acc, file = open(max_acc_log_file, 'a'))
    #
    try:
        os.makedirs(model_log_file)
    except:
        pass
    model_weights_file = os.path.join(model_log_file, exp_name+'_weights.h5')
    model.save_weights(model_weights_file)

    # Clean up the TF session.
    # coord.request_stop()
    # coord.join(threads)
    # K.clear_session()
    
    return LSTM_cell,reshapor,densor

def get_inference_model(FLAGS,LSTM_cell, reshapor, densor):
    
    notes_range = FLAGS.notes_range
    embedding_len = FLAGS.embedding_len 
    layer_size = FLAGS.layer_size
    Ty = FLAGS.Ty

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

    slicer = Lambda(lambda x:x[:,:,notes_range:])
    concator = keras.layers.Concatenate(axis=-1)
    # Define the input of your model with a shape 
    x0 = Input(shape=(embedding_len, notes_range))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(layer_size,), name='a0')
    c0 = Input(shape=(layer_size,), name='c0')
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
        out = reshapor(out)
        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
        outputs.append(out)
        
        # Step 2.D: Select the next value according to "out", and set "x" to be the one-hot representation of the
        #           selected value, which will be passed as the input to LSTM_cell on the next step. We have provided 
        #           the line of code you need to do this. 
        x = slicer(x)
        x = concator([x,out])
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0,a0,c0],outputs=outputs)
    
    ### END CODE HERE ###
    
    return inference_model


if __name__ == '__main__':
    LSTM_cell,reshapor,densor = train(FLAGS)
    inference_model = get_inference_model(FLAGS,LSTM_cell,reshapor,densor)
    inference_model.predict([X_initial,a0,c0],batch_size=predict_batchsize)
      

'''

python3 train_np.py --layer_size=64 \
    --notes_range=38 \
    --batch_size=32 \
    --Ty=100 \
    --epochs=1 \
    --embedding_len=1 \
    --sequence_example_train_file=Wikifonia_basic_rnn_sequence_examples/training_melodies.tfrecord \
    --sequence_example_eval_file=Wikifonia_basic_rnn_sequence_examples/eval_melodies.tfrecord \
    --maxlen=20


'''