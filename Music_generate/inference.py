import keras
from keras.models import load_model,Model
from keras.layers import Dense,Activation,Dropout,Input,LSTM,Reshape,Lambda,RepeatVector

from keras.initializers import glorot_uniform
from keras.utils import to_categorical

from keras.optimizers import Adam
from keras import backend as K

import numpy as np

def get_inferece_model(LSTM_cell,densor,n_values=78, n_a =64, Ty = 100):

    x0 = Input(shape=(1,n_values))

    a0 = Input(shape=(n_a,),name='a0')
    c0 = Input(shape=(n_a,),name='c0')

    a = a0
    c = c0   
    x = x0

    outputs = []

    for t in range(Ty):

        a, _ ,c = LSTM_cell(x,initial_state = [a,c])

        out = densor(a)

        outputs.append(out)

        x = Lambda(onehot)(out)

    inference_model = Model(inputs = [x0,a0,c0],outputs = outputs)

    return inference_model

 