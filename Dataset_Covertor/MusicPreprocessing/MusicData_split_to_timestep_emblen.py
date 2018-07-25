import numpy as np
import xxhash
import time
import io
from sequence_example_lib import *
import os
import tensorflow as tf
sequence_example_file = ('~/sss/Mag/Mag_Data/data_from_WH/Transmit/S_E_BasicRNN_midishare_Mozart20180725/training_melodies.tfrecord')
sequence_example_file_paths = tf.gfile.Glob(
    os.path.expanduser(sequence_example_file))

def get_split_data_emblen_timestep(inputs, labels, time_step = 10, infor_length = 15):
    # cut the text in semi-redundant sequences of time_step characters
    step = 1

    inputs_emb = []
    label_emb = []
    for i in range(0, len(inputs) - infor_length, step):
        inputs_emb.append(inputs[i: i + infor_length].flatten())
        label_emb.append(labels[i + infor_length])

    inputs_time_step = []
    label_time_step = []
    for i in range(0, len(inputs_emb) - time_step, step):
        inputs_time_step.append((inputs_emb[i: i + time_step]))
        label_time_step.append(label_emb[i : i+time_step])

    # return inputs_emb, label_emb
    return inputs_time_step,label_time_step

input_size = 38
inputs, labels, lengths = get_numpy_from_tf_sequence_example( input_size=input_size,
                                    sequence_example_file_paths = sequence_example_file_paths,
                                    shuffle = False)
batch_size = 512
time_step = 10
infor_length = 15
how_much_part = 10
concate_inputs = inputs[:int(len(inputs) / how_much_part)]
concate_labels = labels[:int(len(labels) / how_much_part)]

# infor_length = 9
long_inputs_melodies = []
long_label = []
for index,var in enumerate(inputs):
    inputs_emb, label_emb = get_split_data_emblen_timestep(inputs[index],labels[index],time_step=time_step, infor_length=infor_length)
    long_inputs_melodies.extend(inputs_emb)
    long_label.extend(label_emb)


###Now the long_inputs_melodies and long_label can be fit
#like keras.model.fit(long_inputs_melodies, long_label)

#Analyse how the shape changed:
print('inputs shape',np.shape(inputs))
print('labels shape',np.shape(labels))
print('We let input data fit the emb_len and time_steps, so original input shape ', np.shape(inputs),
      'become','(num_of_music*padded_melody_length, timestep, embed_len*input_size=',
      "(%d*%d, %d, %d*%d)"%(inputs.shape[0],inputs.shape[1],time_step,infor_length,input_size))
print('The label become(num_of_music*padded_melody_length,timestep, label_input_size) = ',
      "(%d*%d, %d, %d)" % (labels.shape[0], labels.shape[1], time_step, len(labels[0]))
      )
print('So the shape become:')
print('long_inputs_melodies shape',np.shape(long_inputs_melodies))
print('long_label shape',np.shape(long_label))

import ipdb; ipdb.set_trace()
