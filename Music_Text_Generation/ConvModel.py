from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential,Model
from keras.layers import Dense, Activation, LSTM, Input, \
    Reshape, MaxPooling1D, Conv1D, Dropout,GlobalMaxPooling1D, \
    LocallyConnected1D,BatchNormalization,AveragePooling1D,Flatten, \
    AtrousConvolution1D, Lambda,merge,concatenate,multiply, UpSampling1D
from keras.optimizers import RMSprop, Adam
import keras
import numpy as np
from keras.regularizers import l2

melody_feature_length = 200

def get_complex_model(input_shape_melody, input_shape_accom,  output_shape):
    note_length = input_shape_melody[1]

    def get_melody_feature(main_melody):
        #Dense Attention or Conv?
        mask = Conv1D(filters=input_shape_melody[1], kernel_size=4,
                   padding='same', activation='sigmoid', strides=1)(main_melody)
        # x = merge([main_melody, mask], output_shape=input_shape_melody[0], name='attention_mul', mode='mul')
        x = multiply([main_melody, mask],name='attention_mul')
        x = LocallyConnected1D(filters=16, kernel_size=32, padding='valid', activation='sigmoid',strides=1)(x)
        x = LocallyConnected1D(filters=32, kernel_size=16, padding='valid', activation='sigmoid', strides=1)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(melody_feature_length, activation='sigmoid')(x)
        return x

    def get_current_melody_feature(before_accom, current_melody):
        before_accom = Conv1D(filters=32, kernel_size=4, padding='same', activation='relu',strides=1)(before_accom)
        before_accom = GlobalMaxPooling1D()(before_accom)

        current_melody = Conv1D(filters=16, kernel_size=8, padding='same', activation='relu',strides=1)(current_melody)
        current_melody = Conv1D(filters=32, kernel_size=4, padding='same', activation='relu',strides=1)(current_melody)
        current_melody = GlobalMaxPooling1D()(current_melody)

        x = concatenate([before_accom,current_melody])
        x = Dense(melody_feature_length, activation='sigmoid')(x)
        return x

    input_melody = Input(shape=input_shape_melody)
    input_accom = Input(shape=input_shape_accom)

    # TODO part of the input_melody
    # import ipdb; ipdb.set_trace()
    input_accom_ = get_current_melody_feature(input_accom,input_melody)
    input_melody_ = get_melody_feature(input_melody)

    x = concatenate([input_accom_,input_melody_])
    x = Dense(melody_feature_length*2, activation='sigmoid')(x)
    x = Dense(output_shape)(x)
    prediction = Activation('softmax')(x)
    model = Model(inputs=[input_melody, input_accom], outputs = prediction)
    return model

def get_complex_model(input_shape_melody, input_shape_accom,  output_shape):
    note_length = input_shape_melody[1]

    def get_melody_feature(main_melody):
        #Dense Attention or Conv?
        mask = Conv1D(filters=input_shape_melody[1], kernel_size=4,
                   padding='same', activation='sigmoid', strides=1)(main_melody)
        # x = merge([main_melody, mask], output_shape=input_shape_melody[0], name='attention_mul', mode='mul')
        x = multiply([main_melody, mask],name='attention_mul')
        x = LocallyConnected1D(filters=16, kernel_size=32, padding='valid', activation='sigmoid',strides=1)(x)
        x = LocallyConnected1D(filters=32, kernel_size=16, padding='valid', activation='sigmoid', strides=1)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(melody_feature_length, activation='sigmoid')(x)
        return x

    def get_current_melody_feature(before_accom, current_melody):
        before_accom = Conv1D(filters=32, kernel_size=4, padding='same', activation='relu',strides=1)(before_accom)
        before_accom = GlobalMaxPooling1D()(before_accom)

        current_melody = Conv1D(filters=16, kernel_size=8, padding='same', activation='relu',strides=1)(current_melody)
        current_melody = Conv1D(filters=32, kernel_size=4, padding='same', activation='relu',strides=1)(current_melody)
        current_melody = GlobalMaxPooling1D()(current_melody)

        x = concatenate([before_accom,current_melody])
        x = Dense(melody_feature_length, activation='sigmoid')(x)
        return x

    input_melody = Input(shape=input_shape_melody)
    input_accom = Input(shape=input_shape_accom)

    # TODO part of the input_melody
    # import ipdb; ipdb.set_trace()
    input_accom_ = get_current_melody_feature(input_accom,input_melody)
    input_melody_ = get_melody_feature(input_melody)

    x = concatenate([input_accom_,input_melody_])
    x = Dense(melody_feature_length*2, activation='sigmoid')(x)
    x = Dense(output_shape)(x)
    prediction = Activation('softmax')(x)
    model = Model(inputs=[input_melody, input_accom], outputs = prediction)
    return model


def get_complex_model_resNet_melody(input_shape_melody, input_shape_accom,  output_shape):
    note_length = input_shape_melody[1]

    def get_melody_feature(main_melody):
        model = get_conv1d_resnet(input_shape_melody,melody_feature_length, input_tensor=main_melody)
        return model.output
    
    def get_current_melody_feature(before_accom, current_melody):
        before_accom = Conv1D(filters=32, kernel_size=4, padding='same', activation='relu',strides=1)(before_accom)
        before_accom = GlobalMaxPooling1D()(before_accom)

        current_melody = Conv1D(filters=16, kernel_size=8, padding='same', activation='relu',strides=1)(current_melody)
        current_melody = Conv1D(filters=32, kernel_size=4, padding='same', activation='relu',strides=1)(current_melody)
        current_melody = GlobalMaxPooling1D()(current_melody)

        x = concatenate([before_accom,current_melody])
        x = Dense(melody_feature_length, activation='sigmoid')(x)
        return x

    input_melody = Input(shape=input_shape_melody)
    input_accom = Input(shape=input_shape_accom)

    # TODO part of the input_melody
    input_accom_ = get_current_melody_feature(input_accom,input_melody)
    input_melody_ = get_melody_feature(input_melody)

    x = concatenate([input_accom_,input_melody_])
    x = Dense(melody_feature_length*2, activation='sigmoid')(x)
    x = Dense(output_shape)(x)
    prediction = Activation('softmax')(x)
    model = Model(inputs=[input_melody, input_accom], outputs = prediction)
    return model


def get_conv1d_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    xxx = Conv1D(filters=16, kernel_size=32, padding='same', activation='sigmoid', strides=1)(inputs)
    xxx2 = Conv1D(filters=16, kernel_size=32, padding='same', activation='sigmoid', strides=1)(inputs)
    xxx = merge([xxx, xxx2], output_shape=32, name='attention_mul', mode='mul')
    xxx = LocallyConnected1D(filters=32, kernel_size=32, padding='valid', activation='sigmoid',strides=1)(xxx)
    xxx = LocallyConnected1D(filters=64, kernel_size=16, padding='valid', activation='sigmoid', strides=1)(xxx)
    # we use max pooling:
    xxx = GlobalMaxPooling1D()(xxx)
    xxx = Dense(output_shape)(xxx)
    predictions = Activation('softmax')(xxx)
    model = Model(inputs=inputs, outputs=predictions)
    return model


def get_lstm_model(input_shape, output_shape, LayerSize=512):
    inputs = Input(shape=input_shape)
    xxx = LSTM(LayerSize, return_sequences=False)(inputs)
    xxx = Dense(output_shape)(xxx)
    predictions = Activation('softmax')(xxx)
    model = Model(inputs=inputs, outputs=predictions)
    return model


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=False,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    # conv = Conv1D(num_filters,
    #               kernel_size=kernel_size,
    #               strides=strides,
    #               padding='same',
    #               kernel_initializer='he_normal',
    #               kernel_regularizer=l2(1e-4))
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  )
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10,input_tensor=None):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    if (input_tensor == None):
        inputs = Input(shape=input_shape)
    else:
        inputs = input_tensor
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             kernel_size=30,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             kernel_size=20,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=10,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling1D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_conv1d_resnet(input_shape, output_shape, input_tensor):
    return resnet_v1(input_shape, num_classes=output_shape, depth=1 * 6 + 2, input_tensor=input_tensor)
    # return resnet_v1(input_shape,num_classes=output_shape, depth=3*6+2, input_tensor=input_tensor)