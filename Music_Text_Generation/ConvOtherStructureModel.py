from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import AveragePooling1D
from keras.layers import GlobalAveragePooling1D, LocallyConnected1D, Lambda
from keras.layers import GlobalMaxPooling1D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K

def conv1d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv1D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv1D`.
        strides: strides in `Conv1D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv1D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    bn_axis = 1
    x = Conv1D(
        filters, (num_row),
        strides=strides[0],
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

#padding_length should be the next conv's filter size
def same_padding_second_dim(x, padding_length, name):
    if(padding_length % 2 == 0 ):
        l , r = padding_length/2 , padding_length/2 - 1
    else:
        l , r = padding_length/2 , padding_length/2 + 1
    l , r = int(l), int(r)
    if(l < 0 ):l = 0
    if(r < 0): r = 0
    x = Lambda(lambda x: K.temporal_padding(x, padding=(l, r)))
    # x = K.temporal_padding(x, padding=(l,r))
    return x


def InceptionV3(include_top=True,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):

    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    img_input = Input(shape=input_shape)
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    channel_axis = 2

    x = conv1d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv1d_bn(x, 32, 3, 3, padding='valid')
    x = conv1d_bn(x, 64, 3, 3)
    x = MaxPooling1D((3), strides=(2))(x)

    x = conv1d_bn(x, 80, 1, 1, padding='valid')
    x = conv1d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling1D((3), strides=(2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv1d_bn(x, 64, 1, 1)

    branch5x5 = conv1d_bn(x, 48, 1, 1)
    branch5x5 = conv1d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv1d_bn(x, 64, 1, 1)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling1D((3), strides=(1), padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv1d_bn(x, 64, 1, 1)

    branch5x5 = conv1d_bn(x, 48, 1, 1)
    branch5x5 = conv1d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv1d_bn(x, 64, 1, 1)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling1D((3), strides=(1), padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv1d_bn(x, 64, 1, 1)

    branch5x5 = conv1d_bn(x, 48, 1, 1)
    branch5x5 = conv1d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv1d_bn(x, 64, 1, 1)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling1D((3), strides=(1), padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv1d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv1d_bn(x, 64, 1, 1)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv1d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling1D((3), strides=(2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv1d_bn(x, 192, 1, 1)

    branch7x7 = conv1d_bn(x, 128, 1, 1)
    branch7x7 = conv1d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv1d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv1d_bn(x, 128, 1, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling1D((3), strides=(1), padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv1d_bn(x, 192, 1, 1)

        branch7x7 = conv1d_bn(x, 160, 1, 1)
        branch7x7 = conv1d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv1d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv1d_bn(x, 160, 1, 1)
        branch7x7dbl = conv1d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv1d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv1d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling1D(
            (3), strides=(1), padding='same')(x)
        branch_pool = conv1d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv1d_bn(x, 192, 1, 1)

    branch7x7 = conv1d_bn(x, 192, 1, 1)
    branch7x7 = conv1d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv1d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv1d_bn(x, 192, 1, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling1D((3), strides=(1), padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv1d_bn(x, 192, 1, 1)
    branch3x3 = conv1d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv1d_bn(x, 192, 1, 1)
    branch7x7x3 = conv1d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv1d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv1d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling1D((3), strides=(2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv1d_bn(x, 320, 1, 1)

        branch3x3 = conv1d_bn(x, 384, 1, 1)
        branch3x3_1 = conv1d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv1d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv1d_bn(x, 448, 1, 1)
        branch3x3dbl = conv1d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv1d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv1d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling1D(
            (3), strides=(1), padding='same')(x)
        branch_pool = conv1d_bn(branch_pool, 192, 1, 1)
        print(branch1x1.shape, branch3x3.shape, branch3x3dbl.shape, branch_pool.shape)
        print('#########'*3)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = GlobalAveragePooling1D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inception_v3')

    return model



def InceptionV3_Local(include_top=True,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    def conv1d_bn_local(x,
                        filters,
                        num_row,
                        num_col,
                        padding='valid',
                        strides=(1, 1),
                        name=None):
        strides = (1, 1)
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        print("****" * 10)
        print('before x:', x.shape, 'num row', num_row)
        x = same_padding_second_dim(x, padding_length=num_row, name=x.name.split('/')[0])(x)
        print('after x:', x.shape)
        bn_axis = 1
        x = LocallyConnected1D(
            filters, (num_row),
            strides=strides[0],
            padding=padding,
            use_bias=False,
            name=conv_name)(x)
        print('after local conv:', x.shape)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Activation('relu', name=name)(x)
        return x

    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    img_input = Input(shape=input_shape)
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 2
    channel_axis = 2

    x = conv1d_bn_local(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv1d_bn_local(x, 32, 3, 3, padding='valid')
    x = conv1d_bn_local(x, 64, 3, 3)
    x = MaxPooling1D((3), strides=(2))(x)

    x = conv1d_bn(x, 80, 1, 1, padding='valid')
    x = conv1d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling1D((3), strides=(2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv1d_bn(x, 64, 1, 1)

    branch5x5 = conv1d_bn(x, 48, 1, 1)
    branch5x5 = conv1d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv1d_bn(x, 64, 1, 1)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling1D((3), strides=(1), padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv1d_bn(x, 64, 1, 1)

    branch5x5 = conv1d_bn(x, 48, 1, 1)
    branch5x5 = conv1d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv1d_bn(x, 64, 1, 1)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling1D((3), strides=(1), padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv1d_bn(x, 64, 1, 1)

    branch5x5 = conv1d_bn(x, 48, 1, 1)
    branch5x5 = conv1d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv1d_bn(x, 64, 1, 1)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling1D((3), strides=(1), padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv1d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv1d_bn(x, 64, 1, 1)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv1d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling1D((3), strides=(2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv1d_bn(x, 192, 1, 1)

    branch7x7 = conv1d_bn(x, 128, 1, 1)
    branch7x7 = conv1d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv1d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv1d_bn(x, 128, 1, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling1D((3), strides=(1), padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv1d_bn(x, 192, 1, 1)

        branch7x7 = conv1d_bn(x, 160, 1, 1)
        branch7x7 = conv1d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv1d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv1d_bn(x, 160, 1, 1)
        branch7x7dbl = conv1d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv1d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv1d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling1D(
            (3), strides=(1), padding='same')(x)
        branch_pool = conv1d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv1d_bn(x, 192, 1, 1)

    branch7x7 = conv1d_bn(x, 192, 1, 1)
    branch7x7 = conv1d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv1d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv1d_bn(x, 192, 1, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling1D((3), strides=(1), padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv1d_bn(x, 192, 1, 1)
    branch3x3 = conv1d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv1d_bn(x, 192, 1, 1)
    branch7x7x3 = conv1d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv1d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv1d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling1D((3), strides=(2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(1):
        # if(i==1): conv1d_bn = conv1d_bn_local
        branch1x1 = conv1d_bn(x, 320, 1, 1)

        branch3x3 = conv1d_bn(x, 384, 1, 1)
        branch3x3_1 = conv1d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv1d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv1d_bn(x, 448, 1, 1)
        branch3x3dbl = conv1d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv1d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv1d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling1D(
            (3), strides=(1), padding='same')(x)
        branch_pool = conv1d_bn(branch_pool, 192, 1, 1)
        print(branch1x1.shape, branch3x3.shape, branch3x3dbl.shape, branch_pool.shape)
        print('#########'*3)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))

    # if(i==1): conv1d_bn = conv1d_bn_local
    i = 1
    conv1d_bn_local = conv1d_bn
    branch1x1 = conv1d_bn_local(x, 320, 2, 1)

    branch3x3 = conv1d_bn_local(x, 384, 2, 1)
    branch3x3_1 = conv1d_bn_local(branch3x3, 384, 2, 3)
    branch3x3_2 = conv1d_bn_local(branch3x3, 384, 4, 1)

    # channel_axis = 1

    branch3x3 = layers.concatenate(
        [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

    branch3x3dbl = conv1d_bn_local(x, 448, 2, 1)
    branch3x3dbl = conv1d_bn_local(branch3x3dbl, 384, 4, 3)
    branch3x3dbl_1 = conv1d_bn_local(branch3x3dbl, 384, 2, 3)
    branch3x3dbl_2 = conv1d_bn_local(branch3x3dbl, 384, 2, 1)
    branch3x3dbl = layers.concatenate(
        [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

    branch_pool = AveragePooling1D(
        (3), strides=(1), padding='same')(x)
    branch_pool = conv1d_bn_local(branch_pool, 192, 2, 1)

    print(branch1x1.shape,branch3x3.shape, branch3x3dbl.shape, branch_pool.shape )
    x = layers.concatenate(
        [branch1x1, branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed' + str(9 + i))




    if include_top:
        # Classification block
        x = GlobalAveragePooling1D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inception_v3')

    return model