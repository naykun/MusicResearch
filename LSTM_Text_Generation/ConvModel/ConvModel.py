from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LSTM, Input, \
    Reshape, MaxPooling1D, Conv1D, Dropout, GlobalMaxPooling1D, \
    LocallyConnected1D, BatchNormalization, AveragePooling1D, Flatten, \
    AtrousConvolution1D, Lambda, merge, concatenate, multiply, UpSampling1D
from keras.optimizers import RMSprop, Adam
import keras
import keras.backend as K
import numpy as np
from keras.regularizers import l2
# from keras_contrib.layers import InstanceNormlization, Weight

melody_feature_length = 100
s_filter_num, s_filter_size = 8 , 8
m_filter_num, m_filter_size = 16, 16
l_filter_num, l_filter_size = 32, 32
xl_filter_num, xl_filter_size = 64, 64
xxl_filter_num, xxl_filter_size = 128, 128
default_activation = 'relu'

### Final
# Accuracy :17epoch 98%, max acc : 1, Weights num : 727,428
def get_conv1d_model_old(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    xxx = inputs

    xxx = LocallyConnected1D(filters=m_filter_num, kernel_size=s_filter_num, padding='valid',
                             activation=default_activation,strides=1)(xxx)
    xxx = LocallyConnected1D(filters=l_filter_num, kernel_size=m_filter_num, padding='valid',
                             activation=default_activation, strides=1)(xxx)
    # we use max pooling:
    xxx = GlobalMaxPooling1D()(xxx)
    xxx = Dense(output_shape)(xxx)
    predictions = Activation('softmax')(xxx)
    model = Model(inputs=inputs, outputs=predictions)
    return model

#### Accuracy :17epoch 98%, max acc : 1, Weights num : 1,505,612
def get_conv1d_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    xxx = inputs

    xxx = Conv1D(filters=s_filter_num, kernel_size=s_filter_size, padding='same', activation=default_activation,
                 strides=1, dilation_rate=1)(xxx)
    xxx = BatchNormalization()(xxx)
    res_x = xxx
    xxx = Conv1D(filters=s_filter_num, kernel_size=s_filter_size, padding='same', activation=default_activation,
                 strides=1, dilation_rate=2)(xxx)
    xxx = BatchNormalization()(xxx)
    xxx = Conv1D(filters=s_filter_num, kernel_size=s_filter_num, padding='same', activation=default_activation,
                 strides=1, dilation_rate=4)(xxx)
    xxx = BatchNormalization()(xxx)
    keras.layers.add([xxx, res_x])
    xxx = LocallyConnected1D(filters=l_filter_num, kernel_size=s_filter_num, padding='valid',
                             activation=default_activation, strides=1)(xxx)
    xxx = LocallyConnected1D(filters=xl_filter_num, kernel_size=m_filter_num, padding='valid',
                             activation=default_activation, strides=1)(xxx)
    xxx = BatchNormalization()(xxx)
    # xxx = Activation(default_activation)(xxx)
    # we use max pooling:
    xxx = GlobalMaxPooling1D()(xxx)
    xxx = Dense(output_shape)(xxx)
    predictions = Activation('softmax')(xxx)
    model = Model(inputs=inputs, outputs=predictions)
    return model

def get_naive_conv1d_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    xxx = inputs

    xxx = Conv1D(filters=l_filter_num, kernel_size=s_filter_num, padding='valid',
                             activation=default_activation,strides=1)(xxx)
    xxx = Conv1D(filters=xl_filter_num, kernel_size=m_filter_num, padding='valid',
                             activation=default_activation, strides=1)(xxx)
    # we use max pooling:
    xxx = GlobalMaxPooling1D()(xxx)
    xxx = Dense(output_shape)(xxx)
    predictions = Activation('softmax')(xxx)
    model = Model(inputs=inputs, outputs=predictions)
    return model

def get_lstm_model(input_shape, output_shape, LayerSize=256):
    inputs = Input(shape=input_shape)
    xxx = LSTM(LayerSize, return_sequences=False)(inputs)
    xxx = Dense(output_shape)(xxx)
    predictions = Activation('softmax')(xxx)
    model = Model(inputs=inputs, outputs=predictions)
    return model


#TODO
def get_two_pipeline_model(input_shape, output_shape, input_unit_length=16):
    def get_melody_feature(main_melody):
        # Dense Attention or Conv?
        mask = Conv1D(filters=input_shape[1], kernel_size=32,
                      padding='same', activation='sigmoid', strides=1, dilation_rate=4)(main_melody)
        # x = merge([main_melody, mask], output_shape=input_shape_melody[0], name='attention_mul', mode='mul')
        x = multiply([main_melody, mask], name='attention_mul')
        x = LocallyConnected1D(filters=16, kernel_size=32, padding='valid', activation='sigmoid', strides=1)(x)
        x = Conv1D(filters=32, kernel_size=32, padding='same', activation='sigmoid', strides=1, dilation_rate=2)(x)
        x = Conv1D(filters=64, kernel_size=32, padding='same', activation='sigmoid', strides=1, dilation_rate=2)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(melody_feature_length, activation='sigmoid')(x)
        return x

    def get_current_melody_feature(current_melody):
        x = LocallyConnected1D(filters=16, kernel_size=8, padding='valid', activation='relu', strides=1)(current_melody)
        x = LocallyConnected1D(filters=32, kernel_size=4, padding='valid', activation='relu', strides=1)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(input_unit_length, activation='sigmoid')(x)
        return x

    input_melody = Input(shape=input_shape)
    all_melody_feature = get_melody_feature(input_melody)
    outs_of_melody_bars = []

    len_of_melody = int(input_melody.shape[1])
    idx = 0
    for i in range(input_unit_length - 1, len_of_melody, input_unit_length):
        outs_of_melody_bars.extend(get_current_melody_feature(input_melody[idx:i]))
        idx = i

    x = concatenate([outs_of_melody_bars, all_melody_feature])
    x = Dense(melody_feature_length * 2, activation='sigmoid')(x)
    x = Dense(output_shape)(x)
    prediction = Activation('softmax')(x)

    model = Model(inputs=input_melody, outputs=prediction)
    return model

def get_complex_model_8_13(input_shape_melody, input_shape_accom, output_shape):
    note_length = input_shape_melody[1]

    def get_melody_feature(main_melody):
        # Dense Attention or Conv?
        mask = Conv1D(filters=input_shape_melody[1], kernel_size=4,
                      padding='same', activation='sigmoid', strides=1)(main_melody)
        # x = merge([main_melody, mask], output_shape=input_shape_melody[0], name='attention_mul', mode='mul')
        x = multiply([main_melody, mask], name='attention_mul')
        x = LocallyConnected1D(filters=16, kernel_size=32, padding='valid', activation='sigmoid', strides=1)(x)
        x = LocallyConnected1D(filters=32, kernel_size=16, padding='valid', activation='sigmoid', strides=1)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(melody_feature_length, activation='sigmoid')(x)
        return x

    def get_current_melody_feature(before_accom, current_melody):
        before_accom = Conv1D(filters=32, kernel_size=4, padding='same', activation='relu', strides=1)(before_accom)
        before_accom = GlobalMaxPooling1D()(before_accom)

        current_melody = Conv1D(filters=16, kernel_size=8, padding='same', activation='relu', strides=1)(current_melody)
        current_melody = Conv1D(filters=32, kernel_size=4, padding='same', activation='relu', strides=1)(current_melody)
        current_melody = GlobalMaxPooling1D()(current_melody)

        x = concatenate([before_accom, current_melody])
        x = Dense(melody_feature_length, activation='sigmoid')(x)
        return x

    input_melody = Input(shape=input_shape_melody)
    input_accom = Input(shape=input_shape_accom)

    # TODO part of the input_melody
    # import ipdb; ipdb.set_trace()
    input_accom_ = get_current_melody_feature(input_accom, input_melody)
    input_melody_ = get_melody_feature(input_melody)

    x = concatenate([input_accom_, input_melody_])
    x = Dense(melody_feature_length * 2, activation='sigmoid')(x)
    x = Dense(output_shape)(x)
    prediction = Activation('softmax')(x)
    model = Model(inputs=[input_melody, input_accom], outputs=prediction)
    return model

def get_complex_model(input_shape_melody, input_shape_accom, output_shape):
    note_length = input_shape_melody[1]

    def get_melody_feature(main_melody):
        # Dense Attention or Conv?
        mask = Conv1D(filters=input_shape_melody[1], kernel_size=32,
                      padding='same', activation='sigmoid', strides=1, dilation_rate=4)(main_melody)
        # x = merge([main_melody, mask], output_shape=input_shape_melody[0], name='attention_mul', mode='mul')
        x = multiply([main_melody, mask], name='attention_mul')
        x = LocallyConnected1D(filters=16, kernel_size=32, padding='valid', activation='sigmoid', strides=1)(x)
        x = Conv1D(filters=32, kernel_size=32, padding='same', activation='sigmoid', strides=1, dilation_rate=2)(x)
        x = Conv1D(filters=64, kernel_size=32, padding='same', activation='sigmoid', strides=1, dilation_rate=2)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(melody_feature_length, activation='sigmoid')(x)
        return x

    def get_current_melody_feature(before_accom, current_melody):
        before_accom = Conv1D(filters=32, kernel_size=4, padding='same', activation='relu', strides=1)(before_accom)
        before_accom = GlobalMaxPooling1D()(before_accom)

        current_melody = Conv1D(filters=16, kernel_size=8, padding='same', activation='relu', strides=1)(current_melody)
        current_melody = Conv1D(filters=32, kernel_size=4, padding='same', activation='relu', strides=1)(current_melody)
        current_melody = GlobalMaxPooling1D()(current_melody)

        x = concatenate([before_accom, current_melody])
        x = Dense(melody_feature_length, activation='sigmoid')(x)
        return x

    input_melody = Input(shape=input_shape_melody)
    input_accom = Input(shape=input_shape_accom)

    # TODO part of the input_melody
    # import ipdb; ipdb.set_trace()
    input_accom_ = get_current_melody_feature(input_accom, input_melody)
    input_melody_ = get_melody_feature(input_melody)

    x = concatenate([input_accom_, input_melody_])
    x = Dense(melody_feature_length * 2, activation='sigmoid')(x)
    x = Dense(output_shape)(x)
    prediction = Activation('softmax')(x)
    model = Model(inputs=[input_melody, input_accom], outputs=prediction)
    return model


# def get_complex_model_resNet_melody(input_shape_melody, input_shape_accom, output_shape):
#     note_length = input_shape_melody[1]
#
#     def get_melody_feature(main_melody):
#        return get_conv1d_resnet(input_shape_melody, melody_feature_length, input_tensor=main_melody)
#
#     def get_current_melody_feature(before_accom, current_melody):
#         before_accom = Conv1D(filters=32, kernel_size=4, padding='same', activation='relu', strides=1)(before_accom)
#         before_accom = GlobalMaxPooling1D()(before_accom)
#
#         current_melody = Conv1D(filters=16, kernel_size=8, padding='same', activation='relu', strides=1)(current_melody)
#         current_melody = Conv1D(filters=32, kernel_size=4, padding='same', activation='relu', strides=1)(current_melody)
#         current_melody = GlobalMaxPooling1D()(current_melody)
#
#         x = concatenate([before_accom, current_melody])
#         x = Dense(melody_feature_length, activation='sigmoid')(x)
#         return x
#
#     input_melody = Input(shape=input_shape_melody)
#     input_accom = Input(shape=input_shape_accom)
#
#     # TODO part of the input_melody
#     input_accom_ = get_current_melody_feature(input_accom, input_melody)
#     input_melody_ = get_melody_feature(input_melody)
#
#     x = concatenate([input_accom_, input_melody_])
#     x = Dense(melody_feature_length * 2, activation='sigmoid')(x)
#     x = Dense(output_shape)(x)
#     prediction = Activation('softmax')(x)
#     model = Model(inputs=[input_melody, input_accom], outputs=prediction)
#     return model




