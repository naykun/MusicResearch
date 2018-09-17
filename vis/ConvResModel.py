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
kernel_size = s_filter_size

#Small filter_size
# s_filter_num, s_filter_size = 8 , 2
# m_filter_num, m_filter_size = 16, 3
# l_filter_num, l_filter_size = 32, 3
# xl_filter_num, xl_filter_size = 64, 3
# xxl_filter_num, xxl_filter_size = 128, 3


### Accuracy :22epoch 99%, max acc : 0.9997593572988782,  weights num: 2,336,932
def resnet_v1_MoreLocal(input_shape, output_shape, depth=3 * 6 + 2):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer_naive(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer_naive(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer_naive(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer_local(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=16,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    x = resnet_layer_local(inputs=x,
                                 num_filters=64,
                                 kernel_size=16,
                                 strides=1,
                                 batch_normalization=True)
    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling1D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(output_shape,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

### Accuracy :23epoch 98%, max acc : 0.9987042315916909, weights num: 501,252
def resnet_v1_simple(input_shape, output_shape, depth=3 * 6 + 2):
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

    inputs = Input(shape=input_shape)
    x = resnet_layer_naive(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer_naive(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer_naive(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer_local(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=16,
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
    outputs = Dense(output_shape,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

### Accuracy :27epoch 98%, max acc : 0.9999, Weights num : 12,409,988
def get_resNet_model(input_shape, output_shape):
    def resnet_v1(input_shape, depth, num_classes=10, input_tensor=None, local_conv=False):
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        if (input_tensor == None):
            inputs = Input(shape=input_shape)
        else:
            inputs = input_tensor

        x = resnet_layer_naive(inputs=inputs)
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
                    x = resnet_layer_naive(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=16,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=True)
                x = keras.layers.add([x, y])
                x = Activation(default_activation)(x)
            num_filters *= 2
        return x

    inputs = Input(shape=input_shape)
    xxx = inputs
    xxx = Conv1D(filters=xl_filter_num, kernel_size=m_filter_num, padding='same',
                 activation=None, strides=1)(xxx)
    xxx = BatchNormalization()(xxx)
    xxx = Activation('relu')(xxx)
    xxx = MaxPooling1D(pool_size=2, padding='same', strides=2)(xxx)

    xxx = resnet_v1(input_shape, num_classes=output_shape, depth=3 * 6 + 2, input_tensor=xxx, local_conv=False)

    xxx = LocallyConnected1D(filters=l_filter_num, kernel_size=m_filter_num, padding='valid',
                             activation=default_activation, strides=1)(xxx)
    xxx = BatchNormalization()(xxx)

    xxx = GlobalMaxPooling1D()(xxx)
    xxx = Dense(output_shape,
                activation='softmax',
                kernel_initializer='he_normal')(xxx)
    model = Model(inputs=inputs, outputs=xxx)
    return model

def get_resnet_model_naive(input_shape, output_shape, depth=3 * 6 + 2):
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

    inputs = Input(shape=input_shape)
    x = resnet_layer_naive(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer_naive(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer_naive(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer_naive(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=16,
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
    outputs = Dense(output_shape,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_layer_naive(inputs,
                 num_filters=16,
                 kernel_size=m_filter_size,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """1D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv1D number of filters
        kernel_size (int): Conv1D square kernel dimensions
        strides (int): Conv1D square stride dimensions
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
                  kernel_regularizer=l2(1e-4))

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

def resnet_layer_local(inputs,
                 num_filters=16,
                 kernel_size=s_filter_size,
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

##TODO
### Accuracy :22epoch 99%, max acc : 0.9997593572988782,  weights num: 2,336,932
def resnet_v1_110(input_shape, output_shape, depth=18 * 6 + 2):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    resnet_layer = resnet_layer_naive
    
    inputs = Input(shape=input_shape)
    x = resnet_layer_naive(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        resnet_layer = resnet_layer_naive
        for res_block in range(num_res_blocks):
            if(res_block == num_res_blocks - 1):
                resnet_layer = resnet_layer_local
            if(stack ==2 and res_block >= num_res_blocks/2):
                resnet_layer = resnet_layer_local
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
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
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    x = resnet_layer_local(inputs=x,
                                 num_filters=64,
                                 kernel_size=16,
                                 strides=1,
                                 batch_normalization=True)
    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling1D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(output_shape,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model