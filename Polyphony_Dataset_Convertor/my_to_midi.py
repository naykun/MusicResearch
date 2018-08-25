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
import copy

from magenta.music import melodies_lib
from magenta.music import melody_encoder_decoder


from sequence_example_lib import *
import os

def get_encoded_events(encoding_config, events_in):
    if(encoding_config == 'basic_rnn'):
        encoding = melody_encoder_decoder.MelodyOneHotEncoding(48, 84)
    else:
        print("Must set encoding_config")
        exit(0)
    events = copy.deepcopy(events_in)
    for i, event in enumerate(events):
        events[i] = encoding.encode_event(event)
    return events

def get_decoded_events(encoding_config, events_in):
    if(encoding_config == 'basic_rnn'):
        encoding = melody_encoder_decoder.MelodyOneHotEncoding(48, 84)
    else:
        print("Must set encoding_config")
        exit(0)
    events = copy.deepcopy(events_in)
    for i, event in enumerate(events):
        events[i] = encoding.decode_event(event)
    return events

def events_to_midi(encoding_config, events_in, output_dir, name):
    """
    events_in: 没有解码过的event_sequence

    """
    events = copy.deepcopy(events_in)
    qpm = magenta.music.DEFAULT_QUARTERS_PER_MINUTE

    # 改变encoding就在这里
    events = get_decoded_events(encoding_config, events)

    melody = melodies_lib.Melody(events)
    sequence = melody.to_sequence(qpm=qpm)

    midi_filename = name+'.mid'
    midi_path = os.path.join(output_dir, midi_filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    magenta.music.sequence_proto_to_midi_file(sequence, midi_path)
    tf.logging.info('Wrote %s.mid to %s', name, output_dir)


'''
def events_to_midi(encoding_config, qpm_in,  events_in, output_dir, name):
    """
    events_in: 没有解码过的event_sequence

    """
    events = copy.deepcopy(events_in)
    qpm = qpm_in

    # 改变encoding就在这里
    events = get_decoded_events(encoding_config, events)

    melody = melodies_lib.Melody(events)
    sequence = melody.to_sequence(qpm=qpm)

    midi_filename = name+'.mid'
    midi_path = os.path.join(output_dir, midi_filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    magenta.music.sequence_proto_to_midi_file(sequence, midi_path)
    tf.logging.info('Wrote %s.mid to %s', name, output_dir)
'''

def get_primer_events(FLAGS):
    """
    return primer melody as event_sequecne, in python-list form

    Args:
        generator: The MelodyRnnSequenceGenerator to use for generation.
        config: The MelodyRnnConfig object
    """
    if FLAGS.primer_melody:
        primer_melody = magenta.music.Melody(ast.literal_eval(FLAGS.primer_melody))
    else:
        tf.logging.warning(
            'No priming sequence specified. Defaulting to a single middle C.')
        primer_melody = magenta.music.Melody([60])

    primer_events = list(primer_melody)

    return primer_events

def one_hot_to_encoded_event_sequence(one_hot_output):
    return list(np.argmax(one_hot_output, axis=1))

# only for basic rnn
def encoded_event_sequence_to_one_hot(encoded_event_sequence, input_size):
    encoded_event_sequence = np.array(encoded_event_sequence)
    one_hot = keras.utils.np_utils.to_categorical(encoded_event_sequence, num_classes=input_size)
    one_hot = np.array(one_hot)
    return one_hot

def output_to_midi(encoding_config, encoded_primer_events, encoded_output_events, output_dir, midi_name):
    if encoded_primer_events == None:
        events_to_midi(encoding_config, encoded_output_events, output_dir, midi_name)
    else:
        output_events = encoded_primer_events + encoded_output_events
        events_to_midi(encoding_config, output_events, output_dir, midi_name)
