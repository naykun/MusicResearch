#coding:utf-8
import keras
import numpy as np
import tensorflow as tf
from sequence_example_lib import *
import os

from train_np import *
from basic_rnn_np_output_to_midi import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('layer_size', 64,
                            'The number of hidden states.')
tf.app.flags.DEFINE_integer('notes_range', 38,
                            'The length of the range of notes.')
tf.app.flags.DEFINE_integer('batch_size', 32,
                            'Batch size.')
tf.app.flags.DEFINE_integer('Ty', 100,
                            'The number of hidden states.')
tf.app.flags.DEFINE_integer('epochs', 1,
                            'Epochs.')
tf.app.flags.DEFINE_integer('embedding_len', 1,
                            'Embedding Length.')
tf.app.flags.DEFINE_string('sequence_example_train_file', '/unsullied/sharefs/ouyangzhihao/DataRoot/AAAI/yk/Wikifonia_basic_rnn_sequence_examples/train/training_melodies.tfrecord',
                           'The directory of sequence example for training.')
tf.app.flags.DEFINE_string('sequence_example_eval_file', '/unsullied/sharefs/ouyangzhihao/DataRoot/AAAI/yk/Wikifonia_basic_rnn_sequence_examples/val/eval_melodies.tfrecord',
                           'The directory of sequence example for validation.')                         
tf.app.flags.DEFINE_integer('maxlen', 200,
                            'max timesteps')

tf.app.flags.DEFINE_string(
    'run_dir', None,
    'Path to the directory where the latest checkpoint will be loaded from.')
tf.app.flags.DEFINE_string(
    'checkpoint_file', None,
    'Path to the checkpoint file. run_dir will take priority over this flag.')
tf.app.flags.DEFINE_string(
    'bundle_file', None,
    'Path to the bundle file. If specified, this will take priority over '
    'run_dir and checkpoint_file, unless save_generator_bundle is True, in '
    'which case both this flag and either run_dir or checkpoint_file are '
    'required')
tf.app.flags.DEFINE_boolean(
    'save_generator_bundle', False,
    'If true, instead of generating a sequence, will save this generator as a '
    'bundle file in the location specified by the bundle_file flag')
tf.app.flags.DEFINE_string(
    'bundle_description', None,
    'A short, human-readable text description of the bundle (e.g., training '
    'data, hyper parameters, etc.).')



tf.app.flags.DEFINE_string(
    'output_dir', 'generated',
    'The directory where MIDI files will be saved to.')
tf.app.flags.DEFINE_integer(
    'num_outputs', 10,
    'The number of melodies to generate. One MIDI file will be created for '
    'each.')
tf.app.flags.DEFINE_string(
    'primer_melody', '',
    'A string representation of a Python list of '
    'magenta.music.Melody event values. For example: '
    '"[60, -2, 60, -2, 67, -2, 67, -2]". If specified, this melody will be '
    'used as the priming melody. If a priming melody is not specified, '
    'melodies will be generated from scratch.')
tf.app.flags.DEFINE_string(
    'primer_midi', '',
    'The path to a MIDI file containing a melody that will be used as a '
    'priming melody. If a primer melody is not specified, melodies will be '
    'generated from scratch.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')



LSTM_cell, reshapor, densor = train(FLAGS)
inference_model = get_inference_model(  FLAGS, 
                                        LSTM_cell,
                                        reshapor,
                                        densor,
                                        notes_range=FLAGS.notes_range, 
                                        embedding_len=FLAGS.embedding_len,
                                        Ty=FLAGS.Ty )


"""Saves bundle or runs generator based on flags."""
tf.logging.set_verbosity(FLAGS.log)

config = melody_rnn_config_flags.config_from_flags()

generator = melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(
      model=melody_rnn_model.MelodyRnnModel(config),
      details=config.details,
      steps_per_quarter=config.steps_per_quarter,
      checkpoint=get_checkpoint()
      )

sequence_example_file = (FLAGS.sequence_example_train_file)
sequence_example_file_paths = tf.gfile.Glob(
    os.path.expanduser(sequence_example_file) )

inputs, labels, lengths = get_numpy_from_tf_sequence_example(   input_size=38,
                                                                sequence_example_file_paths=sequence_example_file_paths,
                                                                shuffle=False   )

print('inputs shape', inputs.shape)
print('inputs type', type(inputs))


primer_events = get_primer_events(FLAGS, generator, config)
print("primer_events:", primer_events)
one_hot_input = encoded_event_sequence_to_one_hot(primer_events, 38)
print("one-hot primer_events shape:", one_hot_input.shape)
print("one-hot primer_events:\n", one_hot_input)

X_initial = one_hot_input




output = inference_model.predict(   FLAGS, 
                                    [X_initial,a0,c0],
                                    batch_size=predict_batchsize    )

encoded_event_sequence = one_hot_to_encoded_event_sequence(output)
event_sequence_to_midi(FLAGS, generator, encoded_event_sequence, 1, config)

'''

python3 train_np.py --layer_size=64 \
    --notes_range=38 \
    --batch_size=32 \
    --Ty=100 \
    --epochs=1 \
    --embedding_len=1 \
    --sequence_example_train_file=Wikifonia_basic_rnn_sequence_examples/train/training_melodies.tfrecord \
    --sequence_example_eval_file=Wikifonia_basic_rnn_sequence_examples/train/eval_melodies.tfrecord \
    --config=basic_rnn \
    --run_dir=logdir/run1 \
    --output_dir=generated \
    --num_outputs=5 \
    --num_steps=10 \
    --hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
    --primer_melody="[60,-2,60,-2,67,-2,67,-2]"




'''