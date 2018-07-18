from sequence_example_lib import *
import os
import tensorflow as tf
sequence_example_file = ('~/sss/Mag/Mag_R/midi_S_E/S_E_AttentionRNN_midishare_Bach/training_melodies.tfrecord')
sequence_example_file_paths = tf.gfile.Glob(
    os.path.expanduser(sequence_example_file))

inputs, labels, lengths = get_numpy_from_tf_sequence_example( batchsize=32, input_size=74,
                                    sequence_example_file_paths = sequence_example_file_paths,
                                    shuffle = False)

print('inputs shape',inputs.shape)
print('inputs type',type(inputs))
print('Finished! And then use ipdb for debug and watch the variables')
import ipdb; ipdb.set_trinputsace()