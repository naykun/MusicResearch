from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LSTM, Input, \
    Reshape, MaxPooling1D, Conv1D, Dropout, GlobalMaxPooling1D, \
    LocallyConnected1D, BatchNormalization, AveragePooling1D, Flatten,\
    AtrousConvolution1D, Lambda, merge, concatenate, multiply, UpSampling1D, concatenate
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


def get_complex_model_resNet_melody(input_shape_melody, input_shape_accom, output_shape):
    note_length = input_shape_melody[1]

    def get_melody_feature(main_melody):
       return get_conv1d_resnet(input_shape_melody, melody_feature_length, input_tensor=main_melody)

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
    input_accom_ = get_current_melody_feature(input_accom, input_melody)
    input_melody_ = get_melody_feature(input_melody)

    x = concatenate([input_accom_, input_melody_])
    x = Dense(melody_feature_length * 2, activation='sigmoid')(x)
    x = Dense(output_shape)(x)
    prediction = Activation('softmax')(x)
    model = Model(inputs=[input_melody, input_accom], outputs=prediction)
    return model


### Final
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

    model = Model(inputs=[input_melody, input_accom], outputs=prediction)
    return model


def get_conv1d_model_naive(input_shape, output_shape):
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
    xxx = Conv1D(filters=l_filter_num, kernel_size=s_filter_num, padding='valid',
                             activation=default_activation, strides=1)(xxx)
    xxx = Conv1D(filters=xl_filter_num, kernel_size=m_filter_num, padding='valid',
                             activation=default_activation, strides=1)(xxx)
    xxx = BatchNormalization()(xxx)
    # xxx = Activation(default_activation)(xxx)
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

#padding_length should be the next conv's filter size
def same_padding_second_dim(x, padding_length, name):
    if(padding_length % 2 == 0 ):
        l , r = padding_length/2 , padding_length/2 - 1
    else:
        l , r = padding_length/2 , padding_length/2 + 1
    l , r = int(l), int(r)

    # x = Lambda(lambda x: K.temporal_padding(x, padding=(l,r)), name='same_padding_second_dim' + str(name))
    x = Lambda(lambda x: K.temporal_padding(x, padding=(l, r)))
    # x = K.temporal_padding(x, padding=(l,r))
    return x

def resnet_layer_naive(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation=default_activation,
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

def resnet_layer_local(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation=default_activation,
                 batch_normalization=False,
                 conv_first=True):

    conv = LocallyConnected1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='valid',
                  kernel_initializer='he_normal',
                  )
    x = inputs
    if conv_first:
        # import ipdb; ipdb.set_trace()
        x = same_padding_second_dim(x, padding_length=kernel_size, name = x.name.split('/')[0])(x)
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
        x = same_padding_second_dim(x, padding_length=kernel_size, name = x.name.split('/')[0])(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10, input_tensor=None, local_conv=False):
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

    if local_conv:
        resnet_layer = resnet_layer_local
    else:
        resnet_layer = resnet_layer_naive
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            # if stack > 0 and res_block == 0:  # first layer but not first stack
            #     strides = 2  # downsample
            y = resnet_layer_local(inputs=x,
                             kernel_size=8,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer_local(inputs=y,
                             kernel_size=16,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=16,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=True)
            x = keras.layers.add([x, y])
            x = Activation(default_activation)(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU

    outputs = x
    # x = MaxPooling1D(pool_size=8)(x)
    # y = Flatten()(x)
    # outputs = Dense(num_classes,
    #                 activation='softmax',
    #                 kernel_initializer='he_normal')(y)

    # Instantiate model.
    if (input_tensor == None):
        return Model(inputs=inputs, outputs=outputs)
    else:
        return outputs
    # model = Model(inputs=inputs, outputs=outputs)
    # return model


def get_conv1d_resnet(input_shape, output_shape, input_tensor):
    return resnet_v1(input_shape, num_classes=output_shape, depth=1 * 6 + 2, input_tensor=input_tensor)
    # return resnet_v1(input_shape,num_classes=output_shape, depth=3*6+2, input_tensor=input_tensor)


#######Get multiple output
import tensorflow as tf
def get_conv1d_model_multiple_out(input_shape, output_shape, output_n):
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
    outs = []
    for i in range(output_n):
        tmp_x = xxx
        tmp_x = Dense(int(output_shape/output_n))(tmp_x)
        predictions = Activation('softmax')(tmp_x)
        outs.append(predictions)
    # outs = tf.Print(outs,[outs],'out softmax')
    # 'concatenate_1'
    predictions = concatenate(outs)

    model = Model(inputs=inputs, outputs=predictions)
    return model

def get_resNet_model_multiple_out(input_shape, output_shape, output_n):
    inputs = Input(shape=input_shape)
    xxx = inputs
    xxx = Conv1D(filters=xl_filter_num, kernel_size=m_filter_num, padding='same',
                             activation=None, strides=1)(xxx)
    xxx = BatchNormalization()(xxx)
    xxx = Activation('relu')(xxx)
    xxx = MaxPooling1D(pool_size=1, padding='same', strides=1)(xxx)

    xxx = resnet_v1(input_shape, num_classes=output_shape, depth=3 * 6 + 2, input_tensor=xxx, local_conv=False)

    xxx = LocallyConnected1D(filters=l_filter_num, kernel_size=m_filter_num, padding='valid',
                             activation=default_activation, strides=1)(xxx)
    xxx = BatchNormalization()(xxx)
    xxx = LocallyConnected1D(filters=l_filter_num, kernel_size=m_filter_num, padding='valid',
                             activation=default_activation, strides=1)(xxx)
    xxx = BatchNormalization()(xxx)
    xxx = LocallyConnected1D(filters=xl_filter_num, kernel_size=4, padding='valid',
                             activation=default_activation, strides=1)(xxx)


    xxx = GlobalMaxPooling1D()(xxx)
    outs = []
    for i in range(output_n):
        tmp_x = xxx
        tmp_x = Dense(int(output_shape/output_n))(tmp_x)
        predictions = Activation('softmax')(tmp_x)
        outs.append(predictions)
    # outs = tf.Print(outs,[outs],'out softmax')
    # 'concatenate_1'
    predictions = concatenate(outs)

    model = Model(inputs=inputs, outputs=predictions)
    return model


def numpy_to_tensor(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = K.constant(y_true)
    y_pred = K.constant(y_pred)
    return y_true, y_pred

def get_perplexity(y_true, y_pred):
    y_true, y_pred = numpy_to_tensor(y_true, y_pred)
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    perplexity = K.pow(2.0, cross_entropy)
    perplexity = K.mean(perplexity)
    perplexity = K.eval(perplexity)
    return perplexity

def get_cross_entropy(y_true, y_pred):
    y_true, y_pred = numpy_to_tensor(y_true, y_pred)
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    cross_entropy = K.mean(cross_entropy)
    cross_entropy = K.eval(cross_entropy)
    return cross_entropy

def get_accuracy(y_true, y_pred):
    acc = K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
    acc = K.eval(acc)
    return acc
