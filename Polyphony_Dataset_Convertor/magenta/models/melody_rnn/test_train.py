from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import six
import tensorflow as tf
import magenta

from tensorflow.python.util import nest as tf_nest

sequence_example_file_paths = "sequence_examples/training_melodies.tfrecord"
hparams_batch_size = 64
input_size =[64,64]

inputs, labels, lengths = None, None, None

inputs, labels, lengths = magenta.common.get_padded_batch(
          sequence_example_file_paths, hparams_batch_size, input_size,
          shuffle=True)
