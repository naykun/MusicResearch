# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate melodies from a trained checkpoint of a melody RNN model."""

import ast
import os
import time

# internal imports

import tensorflow as tf
import numpy as np
import magenta
import keras

import melody_rnn_config_flags
import melody_rnn_model
import melody_rnn_sequence_generator

from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2


from sequence_example_lib import *
import os

def get_checkpoint(FLAGS):
  """Get the training dir or checkpoint path to be used by the model."""
  if FLAGS.run_dir:
    train_dir = os.path.join(os.path.expanduser(FLAGS.run_dir), 'train')
    return train_dir
  

def event_sequence_to_midi(FLAGS, generator, encoded_event_sequence, index, config):
  """

  Uses the given encoded_event_sequence(events start from 0) to write MIDI files.

  Args:
    generator: The MelodyRnnSequenceGenerator to use for generation.
    index: the midi file index
    config: MelodyRnnModelConfig object
  """
  if not FLAGS.output_dir:
    tf.logging.fatal('--output_dir required')
    return
  FLAGS.output_dir = os.path.expanduser(FLAGS.output_dir)

  primer_midi = None
  if FLAGS.primer_midi:
    primer_midi = os.path.expanduser(FLAGS.primer_midi)

  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  primer_sequence = None
  
  
  qpm = magenta.music.DEFAULT_QUARTERS_PER_MINUTE
  

  if FLAGS.primer_melody:
    primer_melody = magenta.music.Melody(ast.literal_eval(FLAGS.primer_melody))
    primer_sequence = primer_melody.to_sequence(qpm=qpm)
  elif primer_midi:
    primer_sequence = magenta.music.midi_file_to_sequence_proto(primer_midi)
    if primer_sequence.tempos and primer_sequence.tempos[0].qpm:
      qpm = primer_sequence.tempos[0].qpm
  else:
    tf.logging.warning(
        'No priming sequence specified. Defaulting to a single middle C.')
    primer_melody = magenta.music.Melody([60])
    primer_sequence = primer_melody.to_sequence(qpm=qpm)

  # Derive the total number of seconds to generate based on the QPM of the
  # priming sequence and the num_steps flag.

  # 一个时间步对应多少秒
  seconds_per_step = 60.0 / qpm / generator.steps_per_quarter
  total_seconds = FLAGS.num_steps * seconds_per_step

  # Specify start/stop time for generation based on starting generation at the
  # end of the priming sequence and continuing until the sequence is num_steps
  # long.
  generator_options = generator_pb2.GeneratorOptions()
  if primer_sequence:
    input_sequence = primer_sequence
    # Set the start time to begin on the next step after the last note ends.
    last_end_time = (max(n.end_time for n in primer_sequence.notes)
                     if primer_sequence.notes else 0)
    generate_section = generator_options.generate_sections.add(
        start_time=last_end_time + seconds_per_step,
        end_time=total_seconds)

    if generate_section.start_time >= generate_section.end_time:
      tf.logging.fatal(
          'Priming sequence is longer than the total number of steps '
          'requested: Priming sequence length: %s, Generation length '
          'requested: %s',
          generate_section.start_time, total_seconds)
      return
  else:
    input_sequence = music_pb2.NoteSequence()
    input_sequence.tempos.add().qpm = qpm
    generate_section = generator_options.generate_sections.add(
        start_time=0,
        end_time=total_seconds)

  # Make the generate request num_outputs times and save the output as midi
  # files.
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
  digits = len(str(FLAGS.num_outputs))

  generated_sequence = generator.event_sequence_to_midi(    input_sequence,
                                                            generator_options,
                                                            encoded_event_sequence,
                                                            config)

  # print("generated_sequence:\n",generated_sequence)

  midi_filename = '%s_%s.mid' % (date_and_time, str(index).zfill(digits) )
  midi_path = os.path.join(FLAGS.output_dir, midi_filename)
  magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)

  tf.logging.info('Wrote %d MIDI files to %s',
                  index, FLAGS.output_dir)


def get_primer_events(FLAGS, generator, config):
  """

  return primer melody as event_sequecne, in python-list form

  Args:
    generator: The MelodyRnnSequenceGenerator to use for generation.
    config: The MelodyRnnConfig object
  """

  FLAGS.output_dir = os.path.expanduser(FLAGS.output_dir)

  primer_midi = None
  if FLAGS.primer_midi:
    primer_midi = os.path.expanduser(FLAGS.primer_midi)

  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  primer_sequence = None
  
  
  qpm = magenta.music.DEFAULT_QUARTERS_PER_MINUTE
  
  
  if FLAGS.primer_melody:
    primer_melody = magenta.music.Melody(ast.literal_eval(FLAGS.primer_melody))
    primer_sequence = primer_melody.to_sequence(qpm=qpm)
  elif primer_midi:
    primer_sequence = magenta.music.midi_file_to_sequence_proto(primer_midi)
    if primer_sequence.tempos and primer_sequence.tempos[0].qpm:
      qpm = primer_sequence.tempos[0].qpm
  else:
    tf.logging.warning(
        'No priming sequence specified. Defaulting to a single middle C.')
    primer_melody = magenta.music.Melody([60])
    primer_sequence = primer_melody.to_sequence(qpm=qpm)

  # Derive the total number of seconds to generate based on the QPM of the
  # priming sequence and the num_steps flag.

  # 一个时间步对应多少秒
  seconds_per_step = 60.0 / qpm / generator.steps_per_quarter
  total_seconds = FLAGS.num_steps * seconds_per_step

  # Specify start/stop time for generation based on starting generation at the
  # end of the priming sequence and continuing until the sequence is num_steps
  # long.
  generator_options = generator_pb2.GeneratorOptions()
  if primer_sequence:
    input_sequence = primer_sequence
    # Set the start time to begin on the next step after the last note ends.
    last_end_time = (max(n.end_time for n in primer_sequence.notes)
                     if primer_sequence.notes else 0)
    generate_section = generator_options.generate_sections.add(
        start_time=last_end_time + seconds_per_step,
        end_time=total_seconds)

    if generate_section.start_time >= generate_section.end_time:
      tf.logging.fatal(
          'Priming sequence is longer than the total number of steps '
          'requested: Priming sequence length: %s, Generation length '
          'requested: %s',
          generate_section.start_time, total_seconds)
      return
  else:
    input_sequence = music_pb2.NoteSequence()
    input_sequence.tempos.add().qpm = qpm
    generate_section = generator_options.generate_sections.add(
        start_time=0,
        end_time=total_seconds)

  primer_events = generator.primer_melody_to_event_sequence(input_sequence,
                                                            generator_options,
                                                            config)

  return primer_events


def one_hot_to_encoded_event_sequence(one_hot_output):
  return list(np.argmax(one_hot_output, axis=1))

# only for basic rnn
def encoded_event_sequence_to_one_hot(encoded_event_sequence, input_size):
  encoded_event_sequence = np.array(encoded_event_sequence)
  one_hot = keras.utils.np_utils.to_categorical(encoded_event_sequence, num_classes=input_size)
  one_hot = np.array(one_hot)
  return one_hot

def main(unused_argv):
  """Saves bundle or runs generator based on flags."""
  tf.logging.set_verbosity('INFO')

  config = melody_rnn_config_flags.config_from_flags()

  generator = melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(
      model=melody_rnn_model.MelodyRnnModel(config),
      details=config.details,
      steps_per_quarter=config.steps_per_quarter,
      checkpoint=get_checkpoint(FLAGS)
      )

  sequence_example_file = ('Wikifonia_basic_rnn_sequence_examples/eval_melodies.tfrecord')
  sequence_example_file_paths = tf.gfile.Glob(
      os.path.expanduser(sequence_example_file))

  inputs, labels, lengths = get_numpy_from_tf_sequence_example(input_size=38,
                                                               sequence_example_file_paths=sequence_example_file_paths,
                                                               shuffle=False)

  print('inputs shape', inputs.shape)
  print('inputs type', type(inputs))


  primer_events = get_primer_events(FLAGS, generator, config)
  print("primer_events:", primer_events)
  one_hot_input = encoded_event_sequence_to_one_hot(primer_events, 38)
  print("one-hot primer_events shape:", one_hot_input.shape)
  print("one-hot primer_events:\n", one_hot_input)

  for i in range(FLAGS.num_outputs):
      test_song = inputs[i]
      encoded_event_sequence = one_hot_to_encoded_event_sequence(test_song)
      event_sequence_to_midi(FLAGS, generator, encoded_event_sequence, i+1, config)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
