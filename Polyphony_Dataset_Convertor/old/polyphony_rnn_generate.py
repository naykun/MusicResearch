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
"""Generate polyphonic tracks from a trained checkpoint.

Uses flags to define operation.
"""

import ast
import os
import time

# internal imports

import tensorflow as tf
import magenta

import polyphony_model
import polyphony_sequence_generator
import polyphony_lib
import polyphony_encoder_decoder

from magenta.music import constants
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'run_dir', None,
    'Path to the directory where the latest checkpoint will be loaded from.')
tf.app.flags.DEFINE_string(
    'bundle_file', None,
    'Path to the bundle file. If specified, this will take priority over '
    'run_dir, unless save_generator_bundle is True, in which case both this '
    'flag and run_dir are required')
tf.app.flags.DEFINE_boolean(
    'save_generator_bundle', False,
    'If true, instead of generating a sequence, will save this generator as a '
    'bundle file in the location specified by the bundle_file flag')
tf.app.flags.DEFINE_string(
    'bundle_description', None,
    'A short, human-readable text description of the bundle (e.g., training '
    'data, hyper parameters, etc.).')
tf.app.flags.DEFINE_string(
    'config', 'polyphony', 'Config to use.')
tf.app.flags.DEFINE_string(
    'output_dir', '/tmp/polyphony_rnn/generated',
    'The directory where MIDI files will be saved to.')
tf.app.flags.DEFINE_integer(
    'num_outputs', 10,
    'The number of tracks to generate. One MIDI file will be created for '
    'each.')
tf.app.flags.DEFINE_integer(
    'num_steps', 128,
    'The total number of steps the generated track should be, priming '
    'track length + generated steps. Each step is a 16th of a bar.')
tf.app.flags.DEFINE_string(
    'primer_pitches', '',
    'A string representation of a Python list of pitches that will be used as '
    'a starting chord with a quarter note duration. For example: '
    '"[60, 64, 67]"')
tf.app.flags.DEFINE_string(
    'primer_melody', '',
    'A string representation of a Python list of '
    'magenta.music.Melody event values. For example: '
    '"[60, -2, 60, -2, 67, -2, 67, -2]".')
tf.app.flags.DEFINE_string(
    'primer_midi', '',
    'The path to a MIDI file containing a polyphonic track that will be used '
    'as a priming track.')
tf.app.flags.DEFINE_boolean(
    'condition_on_primer', False,
    'If set, the RNN will receive the primer as its input before it begins '
    'generating a new sequence.')
tf.app.flags.DEFINE_boolean(
    'inject_primer_during_generation', True,
    'If set, the primer will be injected as a part of the generated sequence. '
    'This option is useful if you want the model to harmonize an existing '
    'melody.')
tf.app.flags.DEFINE_float(
    'qpm', None,
    'The quarters per minute to play generated output at. If a primer MIDI is '
    'given, the qpm from that will override this flag. If qpm is None, qpm '
    'will default to 120.')
tf.app.flags.DEFINE_float(
    'temperature', 1.0,
    'The randomness of the generated tracks. 1.0 uses the unaltered '
    'softmax probabilities, greater than 1.0 makes tracks more random, less '
    'than 1.0 makes tracks less random.')
tf.app.flags.DEFINE_integer(
    'beam_size', 1,
    'The beam size to use for beam search when generating tracks.')
tf.app.flags.DEFINE_integer(
    'branch_factor', 1,
    'The branch factor to use for beam search when generating tracks.')
tf.app.flags.DEFINE_integer(
    'steps_per_iteration', 1,
    'The number of steps to take per beam search iteration.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Comma-separated list of `name=value` pairs. For each pair, the value of '
    'the hyperparameter named `name` is set to `value`. This mapping is merged '
    'with the default hyperparameters.')


def get_checkpoint():
  """Get the training dir or checkpoint path to be used by the model."""
  if FLAGS.run_dir and FLAGS.bundle_file and not FLAGS.save_generator_bundle:
    raise magenta.music.SequenceGeneratorException(
        'Cannot specify both bundle_file and run_dir')
  if FLAGS.run_dir:
    train_dir = os.path.join(os.path.expanduser(FLAGS.run_dir), 'train')
    return train_dir
  else:
    return None


def get_bundle():
  """Returns a generator_pb2.GeneratorBundle object based read from bundle_file.

  Returns:
    Either a generator_pb2.GeneratorBundle or None if the bundle_file flag is
    not set or the save_generator_bundle flag is set.
  """
  if FLAGS.save_generator_bundle:
    return None
  if FLAGS.bundle_file is None:
    return None
  bundle_file = os.path.expanduser(FLAGS.bundle_file)
  return magenta.music.read_bundle_file(bundle_file)


def run_with_flags(generator):
  """Generates polyphonic tracks and saves them as MIDI files.

  Uses the options specified by the flags defined in this module.

  Args:
    generator: The PolyphonyRnnSequenceGenerator to use for generation.
  """
  if not FLAGS.output_dir:
    tf.logging.fatal('--output_dir required')
    return
  output_dir = os.path.expanduser(FLAGS.output_dir)

  primer_midi = None
  if FLAGS.primer_midi:
    primer_midi = os.path.expanduser(FLAGS.primer_midi)

  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  primer_sequence = None
  qpm = FLAGS.qpm if FLAGS.qpm else magenta.music.DEFAULT_QUARTERS_PER_MINUTE
  if FLAGS.primer_pitches:
    primer_sequence = music_pb2.NoteSequence()
    primer_sequence.tempos.add().qpm = qpm
    primer_sequence.ticks_per_quarter = constants.STANDARD_PPQ
    for pitch in ast.literal_eval(FLAGS.primer_pitches):
      note = primer_sequence.notes.add()
      note.start_time = 0
      note.end_time = 60.0 / qpm
      note.pitch = pitch
      note.velocity = 100
    primer_sequence.total_time = primer_sequence.notes[-1].end_time
  elif FLAGS.primer_melody:
    primer_melody = magenta.music.Melody(ast.literal_eval(FLAGS.primer_melody))
    primer_sequence = primer_melody.to_sequence(qpm=qpm)
  elif primer_midi:
    primer_sequence = magenta.music.midi_file_to_sequence_proto(primer_midi)
    if primer_sequence.tempos and primer_sequence.tempos[0].qpm:
      qpm = primer_sequence.tempos[0].qpm
  else:
    tf.logging.warning(
        'No priming sequence specified. Defaulting to empty sequence.')
    primer_sequence = music_pb2.NoteSequence()
    primer_sequence.tempos.add().qpm = qpm
    primer_sequence.ticks_per_quarter = constants.STANDARD_PPQ

  # Derive the total number of seconds to generate.
  seconds_per_step = 60.0 / qpm / generator.steps_per_quarter
  generate_end_time = FLAGS.num_steps * seconds_per_step

  # Specify start/stop time for generation based on starting generation at the
  # end of the priming sequence and continuing until the sequence is num_steps
  # long.
  generator_options = generator_pb2.GeneratorOptions()
  # Set the start time to begin when the last note ends.
  generate_section = generator_options.generate_sections.add(
      start_time=primer_sequence.total_time,
      end_time=generate_end_time)

  if generate_section.start_time >= generate_section.end_time:
    tf.logging.fatal(
        'Priming sequence is longer than the total number of steps '
        'requested: Priming sequence length: %s, Total length '
        'requested: %s',
        generate_section.start_time, generate_end_time)
    return

  generator_options.args['temperature'].float_value = FLAGS.temperature
  generator_options.args['beam_size'].int_value = FLAGS.beam_size
  generator_options.args['branch_factor'].int_value = FLAGS.branch_factor
  generator_options.args[
      'steps_per_iteration'].int_value = FLAGS.steps_per_iteration

  generator_options.args['condition_on_primer'].bool_value = (
      FLAGS.condition_on_primer)
  generator_options.args['no_inject_primer_during_generation'].bool_value = (
      not FLAGS.inject_primer_during_generation)

  tf.logging.debug('primer_sequence: %s', primer_sequence)
  tf.logging.debug('generator_options: %s', generator_options)

  # Make the generate request num_outputs times and save the output as midi
  # files.
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
  digits = len(str(FLAGS.num_outputs))
  for i in range(FLAGS.num_outputs):

    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    pe = polyphony_lib.PolyphonicEvent

    a = [0,2, 2, 71, 2, 70, 2, 71, 2, 73, 2, 74, 2, 66, 2, 67, 2, 69, 2, 67, 2, 64, 2, 76, 2, 74, 2, 73, 2, 71, 2, 70, 2, 79, 2, 78, 2, 76, 2, 74, 2, 73, 2, 71, 2, 70, 2, 71, 2, 73, 2, 71, 2, 73, 2, 74, 2, 74, 73, 73, 2, 74, 2, 73, 73, 71, 2, 74, 2, 73, 2, 71, 59, 2, 73, 58, 2, 74, 59, 2, 76, 61, 2, 78, 62, 2, 206, 54, 2, 79, 78, 55, 2, 78, 57, 2, 76, 55, 2, 204, 52, 2, 64, 2, 62, 2, 76, 61, 2, 204, 59, 2, 77, 76, 58, 2, 76, 67, 2, 74, 66, 2, 202, 64, 2, 62, 2, 61, 2, 73, 59, 2, 201, 58, 2, 74, 59, 2, 202, 61, 2, 67, 59, 2, 195, 61, 2, 195, 62, 2, 64, 62, 61, 2, 69, 62, 61, 61, 2, 64, 59, 2, 66, 62, 61, 2, 194, 61, 2, 74, 59, 2, 202, 61, 2, 202, 62, 2, 73, 64, 2, 74, 66, 2, 69, 194, 2, 71, 67, 66, 2, 74, 66, 2, 79, 64, 2, 78, 192, 2, 76, 2, 74, 2, 73, 64, 2, 71, 192, 2, 69, 66, 64, 2, 73, 64, 2, 78, 62, 2, 76, 190, 2, 74, 2, 73, 2, 71, 61, 2, 69, 189, 2, 67, 62, 2, 71, 190, 2, 76, 55, 2, 71, 183, 2, 73, 183, 2, 201, 52, 2, 201, 57, 2, 73, 52, 2, 74, 54, 2, 69, 182, 2, 71, 62, 2, 69, 190, 2, 78, 190, 2, 206, 61, 2, 206, 62, 2, 206, 57, 2, 206, 59, 2, 69, 62, 2, 71, 67, 2, 69, 66, 2, 79, 64, 2, 207, 62, 2, 207, 61, 2, 207, 59, 2, 207, 57, 2, 69, 61, 2, 74, 66, 2, 73, 64, 2, 78, 62, 2, 76, 61, 2, 79, 59, 2, 78, 57, 2, 83, 55, 2, 81, 59, 2, 79, 64, 2, 78, 62, 2, 76, 62, 61, 2, 81, 61, 2, 79, 189, 2, 81, 61, 2, 78, 62, 2, 81, 57, 2, 79, 59, 2, 81, 57, 2, 74, 66, 2, 78, 194, 2, 76, 194, 2, 78, 194, 2, 71, 194, 2, 73, 57, 2, 74, 59, 2, 76, 57, 2, 73, 67, 2, 74, 195, 2, 76, 195, 2, 78, 195, 2, 74, 195, 2, 76, 57, 2, 78, 62, 2, 79, 61, 2, 81, 66, 2, 79, 64, 2, 83, 67, 2, 81, 66, 2, 79, 71, 2, 78, 69, 2, 76, 67, 2, 79, 66, 2, 73, 64, 2, 201, 69, 2, 201, 67, 2, 201, 69, 2, 201, 66, 2, 73, 194, 2, 78, 50, 2, 73, 52, 2, 74, 54, 2, 71, 56, 2, 73, 57, 2, 69, 49, 2, 71, 50, 2, 199, 52, 2, 199, 50, 2, 199, 47, 2, 199, 59, 2, 78, 57, 2, 77, 56, 2, 80, 54, 2, 71, 53, 2, 69, 62, 2, 71, 61, 2, 199, 59, 2, 199, 57, 2, 77, 56, 2, 75, 54, 2, 73, 53, 2, 81, 54, 2, 209, 56, 2, 209, 54, 2, 75, 56, 2, 77, 57, 56, 2, 205, 57, 2, 205, 57, 56, 2, 78, 56, 54, 2, 78, 57, 56, 2, 206, 56, 2, 66, 54, 2, 65, 56, 2, 66, 57, 2, 68, 59, 2, 69, 61, 2, 189, 61, 2, 62, 61, 2, 64, 189, 2, 62, 59, 2, 187, 59, 2, 71, 2, 69, 2, 68, 59, 2, 66, 187, 2, 65, 61, 59, 2, 74, 59, 2, 73, 57, 2, 71, 185, 2, 69, 2, 68, 2, 66, 56, 2, 65, 184, 2, 66, 57, 2, 68, 185, 2, 66, 50, 2, 68, 178, 2, 69, 178, 2, 69, 68, 47, 2, 69, 68, 52, 2, 68, 68, 66, 47, 2, 69, 49, 2, 68, 177, 2, 66, 57, 2, 68, 185, 2, 69, 185, 2, 71, 56, 2, 73, 57, 2, 201, 52, 2, 74, 73, 54, 2, 73, 57, 2, 71, 62, 2, 199, 61, 2, 59, 2, 57, 2, 71, 56, 2, 199, 54, 2, 73, 71, 52, 2, 71, 56, 2, 69, 61, 2, 197, 59, 2, 57, 2, 56, 2, 68, 54, 2, 196, 52, 2, 69, 50, 2, 197, 54, 2, 62, 59, 2, 190, 54, 2, 190, 57, 56, 2, 59, 56, 2, 64, 184, 2, 59, 56, 2, 61, 57, 2, 189, 52, 2, 69, 54, 2, 197, 52, 2, 197, 61, 2, 68, 189, 2, 69, 189, 2, 64, 189, 2, 66, 189, 2, 69, 52, 2, 74, 54, 2, 73, 52, 2, 71, 62, 2, 69, 190, 2, 68, 190, 2, 66, 190, 2, 64, 190, 2, 68, 52, 2, 73, 57, 2, 71, 56, 2, 69, 61, 2, 68, 59, 2, 66, 62, 2, 64, 61, 2, 66, 62, 2, 66, 64, 2, 71, 62, 2, 66, 61, 2, 68, 59, 2, 196, 64, 2, 196, 62, 2, 68, 64, 2, 69, 61, 2, 64, 2, 66, 62, 2, 64, 2, 73, 57, 2, 201, 61, 2, 201, 59, 2, 201, 61, 2, 201, 54, 2, 64, 56, 2, 66, 57, 2, 64, 59, 2, 74, 56, 2, 202, 57, 2, 202, 59, 2, 202, 61, 2, 202, 57, 2, 64, 59, 2, 69, 61, 2, 68, 62, 2, 73, 64, 2, 71, 62, 2, 74, 66, 2, 73, 64, 2, 78, 62, 2, 76, 61, 2, 74, 59, 2, 73, 62, 2, 71, 56, 2, 76, 184, 2, 74, 184, 2, 76, 184, 2, 73, 184, 2, 71, 49, 2, 69, 54, 2, 68, 53, 2, 73, 57, 2, 71, 56, 2, 74, 59, 2, 73, 57, 2, 71, 62, 2, 69, 61, 2, 68, 59, 2, 71, 57, 2, 76, 56, 2, 204, 61, 2, 204, 59, 2, 204, 61, 2, 204, 58, 2, 66, 64, 2, 71, 62, 2, 70, 61, 2, 74, 59, 2, 73, 57, 2, 76, 55, 2, 74, 54, 2, 79, 52, 2, 78, 50, 2, 76, 49, 2, 74, 47, 2, 73, 54, 2, 78, 182, 2, 76, 42, 2, 78, 170, 2, 74, 47, 2, 73, 175, 2, 71, 175, 2, 70, 49, 2, 71, 50, 2, 73, 52, 2, 74, 54, 2, 66, 182, 2, 67, 55, 54, 2, 69, 54, 2, 67, 52, 2, 64, 180, 2, 76, 2, 74, 2, 73, 52, 2, 71, 180, 2, 70, 54, 52, 2, 79, 52, 2, 78, 50, 2, 76, 178, 2, 74, 178, 2, 73, 52, 2, 71, 50, 2, 70, 49, 2, 71, 47, 2, 73, 45, 2, 71, 43, 2, 73, 42, 2, 74, 40, 2, 74, 73, 38, 2, 74, 73, 40, 2, 73, 71, 42, 2, 74, 73, 35, 2, 73, 163, 2, 71, 47, 2, 73, 46, 2, 74, 47, 2, 76, 49, 2, 78, 50, 2, 206, 42, 2, 79, 78, 43, 2, 78, 45, 2, 76, 43, 2, 204, 40, 2, 52, 2, 50, 2, 76, 49, 2, 204, 47, 2, 78, 76, 46, 2, 76, 55, 2, 74, 54, 2, 73, 52, 2, 78, 50, 2, 76, 49, 2, 79, 47, 2, 78, 46, 2, 78, 47, 2, 206, 42, 2, 206, 43, 2, 79, 40, 2, 74, 73, 42, 2, 74, 73, 73, 170, 2, 74, 73, 42, 2, 71, 170, 2, 71, 35, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163,2,1]
    
    poly_enc_dec = polyphony_encoder_decoder.PolyphonyOneHotEncoding()

    poly_events = []
    for i in a:
      poly_events.append(poly_enc_dec.decode_event(i))
        
    '''
    poly_events = [
        # step 0
        pe(pe.NEW_NOTE, 60),
        pe(pe.END, None),  # END event before end. Should be ignored.
        pe(pe.NEW_NOTE, 64),
        pe(pe.END, None),  # END event before end. Should be ignored.
        pe(pe.STEP_END, None),
        pe(pe.END, None),  # END event before end. Should be ignored.
        # step 1
        pe(pe.CONTINUED_NOTE, 60),
        pe(pe.END, None),  # END event before end. Should be ignored.
        pe(pe.CONTINUED_NOTE, 64),
        pe(pe.END, None),  # END event before end. Should be ignored.
        pe(pe.STEP_END, None),

        pe(pe.END, None),
    ]
    '''
    for event in poly_events:
      poly_seq.append(event)

    poly_seq_ns = poly_seq.to_sequence(qpm=240.0)

    generated_sequence = poly_seq_ns

    # generated_sequence = generator.generate(primer_sequence, generator_options)

    # import ipdb; ipdb.set_trace();

    midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
    midi_path = os.path.join(output_dir, midi_filename)
    magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)

  tf.logging.info('Wrote %d MIDI files to %s',
                  FLAGS.num_outputs, output_dir)


def main(unused_argv):
  """Saves bundle or runs generator based on flags."""
  tf.logging.set_verbosity(FLAGS.log)

  bundle = get_bundle()

  config_id = bundle.generator_details.id if bundle else FLAGS.config
  config = polyphony_model.default_configs[config_id]
  config.hparams.parse(FLAGS.hparams)
  # Having too large of a batch size will slow generation down unnecessarily.
  config.hparams.batch_size = min(
      config.hparams.batch_size, FLAGS.beam_size * FLAGS.branch_factor)

  generator = polyphony_sequence_generator.PolyphonyRnnSequenceGenerator(
      model=polyphony_model.PolyphonyRnnModel(config),
      details=config.details,
      steps_per_quarter=config.steps_per_quarter,
      checkpoint=get_checkpoint(),
      bundle=bundle)

  if FLAGS.save_generator_bundle:
    bundle_filename = os.path.expanduser(FLAGS.bundle_file)
    if FLAGS.bundle_description is None:
      tf.logging.warning('No bundle description provided.')
    tf.logging.info('Saving generator bundle to %s', bundle_filename)
    generator.create_bundle_file(bundle_filename, FLAGS.bundle_description)
  else:
    run_with_flags(generator)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
