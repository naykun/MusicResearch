#coding:utf-8
import keras
import numpy as np
import tensorflow as tf
from sequence_example_lib import *
import os

# from my_to_midi import *
# from basic_rnn_np_output_to_midi import *

from my_to_midi import *
from my_train import *

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
tf.app.flags.DEFINE_string('sequence_example_train_file', 'Wikifonia_basic_rnn_sequence_examples/training_melodies.tfrecord',
                           'The directory of sequence example for training.')
tf.app.flags.DEFINE_string('sequence_example_eval_file', 'Wikifonia_basic_rnn_sequence_examples/eval_melodies.tfrecord',
                           'The directory of sequence example for validation.')                         
tf.app.flags.DEFINE_integer('maxlen', 20,
                            'max timesteps')
tf.app.flags.DEFINE_integer('predict_batch_size', 1,
                            'Predict batch size.')


tf.app.flags.DEFINE_string(
    'output_dir', 'generated',
    'The directory where MIDI files will be saved to.')
tf.app.flags.DEFINE_integer(
    'num_outputs', 10,
    'The number of melodies to generate. One MIDI file will be created for '
    'each.')
tf.app.flags.DEFINE_integer(
    'num_steps', 128,
    'The total number of steps the generated melodies should be, priming '
    'melody length + generated steps. Each step is a 16th of a bar.')
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


LSTM_cell, reshapor, densor = train(FLAGS)
inference_model = get_inference_model(FLAGS, LSTM_cell, reshapor, densor)




tf.logging.set_verbosity('INFO')

config = melody_rnn_config_flags.config_from_flags()

generator = melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(
    model=melody_rnn_model.MelodyRnnModel(config),
    details=config.details,
    steps_per_quarter=config.steps_per_quarter,
    checkpoint=None
    )

sequence_example_file = (FLAGS.sequence_example_train_file)
sequence_example_file_paths = tf.gfile.Glob(os.path.expanduser(sequence_example_file))

primer_events = get_primer_events(FLAGS, generator, config)
print("primer_events:", primer_events)
one_hot_input = encoded_event_sequence_to_one_hot(primer_events, FLAGS.notes_range)
print("one-hot primer_events shape:", one_hot_input.shape)
print("one-hot primer_events:\n", one_hot_input)




X_initial = np.array([[one_hot_input[0]]])
print(X_initial.shape)
# reshape(FLAGS.embedding_len, FLAGS.notes_range)
a0 = np.zeros( (1,FLAGS.layer_size), dtype = int)
c0 = np.zeros( (1,FLAGS.layer_size), dtype = int)

model_output = inference_model.predict([X_initial,a0,c0],batch_size=FLAGS.predict_batch_size)


def model_output_to_one_hot_output(output):
    ret = []
    for i in output:
        ret.append(list(i[0][0]))

    ret = np.array(ret)

    print("model output:", list(ret) )

    # ret = np.rint(ret)
    return ret

output = model_output_to_one_hot_output(model_output)

# import ipdb; ipdb.set_trace()


encoded_event_sequence = one_hot_to_encoded_event_sequence(output)

print("encoded event sequence", encoded_event_sequence)

event_sequence_to_midi(FLAGS, generator, encoded_event_sequence, 1, config)


'''


python3 main.py --layer_size=64 \
    --notes_range=38 \
    --batch_size=64 \
    --predict_batch_size=1 \
    --Ty=100 \
    --epochs=1 \
    --embedding_len=1 \
    --sequence_example_train_file=/home/ouyangzhihao/sss/AAAI/yyh/Wikifonia_basic_rnn_sequence_examples/training_melodies.tfrecord \
    --sequence_example_eval_file=/home/ouyangzhihao/sss/AAAI/yyh/Wikifonia_basic_rnn_sequence_examples/eval_melodies.tfrecord \
    --maxlen=10 \
    --config=basic_rnn \
    --output_dir=generated \
    --num_outputs=5 \
    --num_steps=10 \
    --primer_melody="[60,-2,60,-2,67,-2,67,-2]"


'''


