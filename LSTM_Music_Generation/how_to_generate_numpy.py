from sequence_example_lib import *
import os
import tensorflow as tf
import time
sequence_example_file = ('~/sss/Mag/Mag_Data/data_from_WH/Transmit/S_E_BasicRNN_midishare_bach20180725/training_melodies.tfrecord')
sequence_example_file_paths = tf.gfile.Glob(
    os.path.expanduser(sequence_example_file))

start_time = time.time()
inputs, labels, lengths = get_numpy_from_tf_sequence_example( input_size=38,
                                    sequence_example_file_paths = sequence_example_file_paths,
                                    shuffle = False)
print('Time:',time.time() - start_time)
print('inputs shape',inputs.shape)
print('inputs type',type(inputs))
print('Finished! And then use ipdb for debug and watch the variables')
import ipdb; ipdb.set_trinputsace()