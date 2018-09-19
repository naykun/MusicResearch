
# coding: utf-8

# In[3]:


import ast
import os
import time

# internal imports
import pickle as pkl

import tensorflow as tf
import magenta

import polyphony_lib
import polyphony_encoder_decoder

from  sequence_example_lib import *

from magenta.music import constants
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2

import magenta.music as mm
import numpy as np

# In[ ]:


def to_events(x):
    return np.argmax(x, axis=1)
def to_real_length(x):
    while(x[len(x)-1]==0):
        x.pop()
    # delete the last note along with padded zeros
    x.pop()


def sequence_example_to_real_inputs(sequence_example_path):
    # train
    sequence_example_file = sequence_example_path
    sequence_example_file_paths = tf.gfile.Glob(
        os.path.expanduser(sequence_example_file))
    start_time = time.time()
    inputs, labels, lengths = get_numpy_from_tf_sequence_example( input_size=259,
                                        sequence_example_file_paths = sequence_example_file_paths,
                                        shuffle = False)
    print('Time:',time.time() - start_time)
    print('inputs shape',inputs.shape)
    print('inputs type',type(inputs))
    input_events = []
    for i in inputs:
        input_events.append(to_events(i))
    real_inputs = []
    for i in input_events:
        d = []
        d = list(i)
        to_real_length(d)
        real_inputs.append(d)
    return real_inputs


def polyphony_sequence_example_to_list_pkl(sequence_example_path, output_dir, name):
    real_inputs = sequence_example_to_real_inputs(sequence_example_path)
    real_inputs_path = os.path.join(output_dir, name+'.pkl')
    with open(real_inputs_path, 'wb') as mf:   #pickle只能以二进制格式存储数据到文件
        mf.write(pkl.dumps(real_inputs) )   #dumps序列化源数据后写入文件
        mf.close()
        print("Write %s to %s" % (real_inputs_path, output_dir) )


def midi_to_polyphony_sequence(steps_per_quarter, qpm, midi_dir, midi_name):
    primer_midi = os.path.join(midi_dir, midi_name)
    primer_midi = os.path.expanduser(primer_midi)

    primer_sequence = magenta.music.midi_file_to_sequence_proto(primer_midi)
    if primer_sequence.tempos and primer_sequence.tempos[0].qpm:
        qpm = primer_sequence.tempos[0].qpm
        
    # Quantize the priming sequence.
    quantized_primer_sequence = mm.quantize_note_sequence(
        primer_sequence, steps_per_quarter)

    extracted_seqs, _ = polyphony_lib.extract_polyphonic_sequences(
        quantized_primer_sequence, start_step=0)
    
    poly_seq = extracted_seqs[0]
    
    return poly_seq


def polyphony_sequence_to_list(poly_seq):
    ret = []
    poly_enc_dec = polyphony_encoder_decoder.PolyphonyOneHotEncoding()
    for poly_event in poly_seq:
        ret.append(poly_enc_dec.encode_event(poly_event))
    return ret



def list_to_midi(events, qpm, output_dir, midi_name):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=4)
    pe = polyphony_lib.PolyphonicEvent
    poly_enc_dec = polyphony_encoder_decoder.PolyphonyOneHotEncoding()
    poly_events = []
    for event in events:
        poly_events.append(poly_enc_dec.decode_event(event))
    
    for event in poly_events:
        poly_seq.append(event)

    print(len(poly_events))
    print(len(poly_seq))
    # import ipdb; ipdb.set_trace()
    poly_seq_ns = poly_seq.to_sequence(qpm=qpm)
    
    midi_path = os.path.join(output_dir, midi_name+'.mid')
    magenta.music.sequence_proto_to_midi_file(poly_seq_ns, midi_path)

    print('Wrote %s.mid to %s'%(midi_name, output_dir) )




# polyphony sequence example to list pkl
# poly_S_E_path = '/home/ouyangzhihao/sss/AAAI/common/Mag_Data/poly_midi_S_E/Wikiofnia/training_poly_tracks.tfrecord'
# polyphony_sequence_example_to_list_pkl(poly_S_E_path, '/home/ouyangzhihao/sss/AAAI/common/Mag_Data/Poly_List_Datasets/Wikifonia', 'Wikifonia_poly_train')

dataset_dir = '/home/ouyangzhihao/sss/AAAI/common/Mag_Data/Poly_List_Datasets/Wikifonia'
dataset_name = 'Wikifonia'
train_dataset_path = os.path.join(dataset_dir, dataset_name + '_train.pkl')
eval_dataset_path = os.path.join(dataset_dir, dataset_name + '_eval.pkl')

dataset_new_name = 'Wikifonia_new'
train_dataset_new_path = os.path.join(dataset_dir, dataset_new_name + '_train.pkl')
eval_dataset_new_path = os.path.join(dataset_dir, dataset_new_name + '_eval.pkl')

with open(train_dataset_path, "rb") as train_file:
    train_data = pkl.load(train_file)
    temp = []

    temp = np.concatenate(train_data)

    temp = temp[temp != 0]
    temp = temp[temp != 1]

    # print(temp[0])
    # print(temp[0:100])

    # temp = temp[temp != 0 and temp != 1]

    '''
    for i in train_data:
        cnt += 1
        print(cnt)
        temp = temp + i[1:len(i) - 1]
    '''
    with open(train_dataset_new_path, 'wb') as wf:  # pickle只能以二进制格式存储数据到文件
        wf.write(pkl.dumps(temp))  # dumps序列化源数据后写入文件
        wf.close()

    train_data = np.array(temp)
    train_file.close()

print('Train dataset shape:', train_data.shape)

with open(eval_dataset_path, "rb") as eval_file:
    eval_data = pkl.load(eval_file)
    temp = []

    temp = np.concatenate(eval_data)

    temp = temp[temp != 0]
    temp = temp[temp != 1]

    '''
    for i in eval_data:
        temp = temp + i[1:len(i) - 1]
    '''

    with open(eval_dataset_new_path, 'wb') as wf:  # pickle只能以二进制格式存储数据到文件
        wf.write(pkl.dumps(temp))  # dumps序列化源数据后写入文件
        wf.close()

    eval_data = np.array(temp)
    eval_file.close()

print('Eval dataset shape:', eval_data.shape)

'''
# how to use

# midi to polyphony sequence
test_poly = midi_to_polyphony_sequence(4, 160, '/Users/mac/Desktop/test', 'test0.mid')

# polyphony sequence to list
test_list = polyphony_sequence_to_list(test_poly)
print(test_list)


# polyphony sequence example to list pkl
poly_S_E_path = '/Users/mac/Desktop/MusicGeneration/Mag_Data/Poly_S_E/eval_poly_tracks.tfrecord'
polyphony_sequence_example_to_list_pkl(poly_S_E_path, '/Users/mac/Desktop/test', 'poly_S_E_to_list')


# list to midi
test_events = [0,2, 2, 71, 2, 70, 2, 71, 2, 73, 2, 74, 2, 66, 2, 67, 2, 69, 2, 67, 2, 64, 2, 76, 2, 74, 2, 73, 2, 71, 2, 70, 2, 79, 2, 78, 2, 76, 2, 74, 2, 73, 2, 71, 2, 70, 2, 71, 2, 73, 2, 71, 2, 73, 2, 74, 2, 74, 73, 73, 2, 74, 2, 73, 73, 71, 2, 74, 2, 73, 2, 71, 59, 2, 73, 58, 2, 74, 59, 2, 76, 61, 2, 78, 62, 2, 206, 54, 2, 79, 78, 55, 2, 78, 57, 2, 76, 55, 2, 204, 52, 2, 64, 2, 62, 2, 76, 61, 2, 204, 59, 2, 77, 76, 58, 2, 76, 67, 2, 74, 66, 2, 202, 64, 2, 62, 2, 61, 2, 73, 59, 2, 201, 58, 2, 74, 59, 2, 202, 61, 2, 67, 59, 2, 195, 61, 2, 195, 62, 2, 64, 62, 61, 2, 69, 62, 61, 61, 2, 64, 59, 2, 66, 62, 61, 2, 194, 61, 2, 74, 59, 2, 202, 61, 2, 202, 62, 2, 73, 64, 2, 74, 66, 2, 69, 194, 2, 71, 67, 66, 2, 74, 66, 2, 79, 64, 2, 78, 192, 2, 76, 2, 74, 2, 73, 64, 2, 71, 192, 2, 69, 66, 64, 2, 73, 64, 2, 78, 62, 2, 76, 190, 2, 74, 2, 73, 2, 71, 61, 2, 69, 189, 2, 67, 62, 2, 71, 190, 2, 76, 55, 2, 71, 183, 2, 73, 183, 2, 201, 52, 2, 201, 57, 2, 73, 52, 2, 74, 54, 2, 69, 182, 2, 71, 62, 2, 69, 190, 2, 78, 190, 2, 206, 61, 2, 206, 62, 2, 206, 57, 2, 206, 59, 2, 69, 62, 2, 71, 67, 2, 69, 66, 2, 79, 64, 2, 207, 62, 2, 207, 61, 2, 207, 59, 2, 207, 57, 2, 69, 61, 2, 74, 66, 2, 73, 64, 2, 78, 62, 2, 76, 61, 2, 79, 59, 2, 78, 57, 2, 83, 55, 2, 81, 59, 2, 79, 64, 2, 78, 62, 2, 76, 62, 61, 2, 81, 61, 2, 79, 189, 2, 81, 61, 2, 78, 62, 2, 81, 57, 2, 79, 59, 2, 81, 57, 2, 74, 66, 2, 78, 194, 2, 76, 194, 2, 78, 194, 2, 71, 194, 2, 73, 57, 2, 74, 59, 2, 76, 57, 2, 73, 67, 2, 74, 195, 2, 76, 195, 2, 78, 195, 2, 74, 195, 2, 76, 57, 2, 78, 62, 2, 79, 61, 2, 81, 66, 2, 79, 64, 2, 83, 67, 2, 81, 66, 2, 79, 71, 2, 78, 69, 2, 76, 67, 2, 79, 66, 2, 73, 64, 2, 201, 69, 2, 201, 67, 2, 201, 69, 2, 201, 66, 2, 73, 194, 2, 78, 50, 2, 73, 52, 2, 74, 54, 2, 71, 56, 2, 73, 57, 2, 69, 49, 2, 71, 50, 2, 199, 52, 2, 199, 50, 2, 199, 47, 2, 199, 59, 2, 78, 57, 2, 77, 56, 2, 80, 54, 2, 71, 53, 2, 69, 62, 2, 71, 61, 2, 199, 59, 2, 199, 57, 2, 77, 56, 2, 75, 54, 2, 73, 53, 2, 81, 54, 2, 209, 56, 2, 209, 54, 2, 75, 56, 2, 77, 57, 56, 2, 205, 57, 2, 205, 57, 56, 2, 78, 56, 54, 2, 78, 57, 56, 2, 206, 56, 2, 66, 54, 2, 65, 56, 2, 66, 57, 2, 68, 59, 2, 69, 61, 2, 189, 61, 2, 62, 61, 2, 64, 189, 2, 62, 59, 2, 187, 59, 2, 71, 2, 69, 2, 68, 59, 2, 66, 187, 2, 65, 61, 59, 2, 74, 59, 2, 73, 57, 2, 71, 185, 2, 69, 2, 68, 2, 66, 56, 2, 65, 184, 2, 66, 57, 2, 68, 185, 2, 66, 50, 2, 68, 178, 2, 69, 178, 2, 69, 68, 47, 2, 69, 68, 52, 2, 68, 68, 66, 47, 2, 69, 49, 2, 68, 177, 2, 66, 57, 2, 68, 185, 2, 69, 185, 2, 71, 56, 2, 73, 57, 2, 201, 52, 2, 74, 73, 54, 2, 73, 57, 2, 71, 62, 2, 199, 61, 2, 59, 2, 57, 2, 71, 56, 2, 199, 54, 2, 73, 71, 52, 2, 71, 56, 2, 69, 61, 2, 197, 59, 2, 57, 2, 56, 2, 68, 54, 2, 196, 52, 2, 69, 50, 2, 197, 54, 2, 62, 59, 2, 190, 54, 2, 190, 57, 56, 2, 59, 56, 2, 64, 184, 2, 59, 56, 2, 61, 57, 2, 189, 52, 2, 69, 54, 2, 197, 52, 2, 197, 61, 2, 68, 189, 2, 69, 189, 2, 64, 189, 2, 66, 189, 2, 69, 52, 2, 74, 54, 2, 73, 52, 2, 71, 62, 2, 69, 190, 2, 68, 190, 2, 66, 190, 2, 64, 190, 2, 68, 52, 2, 73, 57, 2, 71, 56, 2, 69, 61, 2, 68, 59, 2, 66, 62, 2, 64, 61, 2, 66, 62, 2, 66, 64, 2, 71, 62, 2, 66, 61, 2, 68, 59, 2, 196, 64, 2, 196, 62, 2, 68, 64, 2, 69, 61, 2, 64, 2, 66, 62, 2, 64, 2, 73, 57, 2, 201, 61, 2, 201, 59, 2, 201, 61, 2, 201, 54, 2, 64, 56, 2, 66, 57, 2, 64, 59, 2, 74, 56, 2, 202, 57, 2, 202, 59, 2, 202, 61, 2, 202, 57, 2, 64, 59, 2, 69, 61, 2, 68, 62, 2, 73, 64, 2, 71, 62, 2, 74, 66, 2, 73, 64, 2, 78, 62, 2, 76, 61, 2, 74, 59, 2, 73, 62, 2, 71, 56, 2, 76, 184, 2, 74, 184, 2, 76, 184, 2, 73, 184, 2, 71, 49, 2, 69, 54, 2, 68, 53, 2, 73, 57, 2, 71, 56, 2, 74, 59, 2, 73, 57, 2, 71, 62, 2, 69, 61, 2, 68, 59, 2, 71, 57, 2, 76, 56, 2, 204, 61, 2, 204, 59, 2, 204, 61, 2, 204, 58, 2, 66, 64, 2, 71, 62, 2, 70, 61, 2, 74, 59, 2, 73, 57, 2, 76, 55, 2, 74, 54, 2, 79, 52, 2, 78, 50, 2, 76, 49, 2, 74, 47, 2, 73, 54, 2, 78, 182, 2, 76, 42, 2, 78, 170, 2, 74, 47, 2, 73, 175, 2, 71, 175, 2, 70, 49, 2, 71, 50, 2, 73, 52, 2, 74, 54, 2, 66, 182, 2, 67, 55, 54, 2, 69, 54, 2, 67, 52, 2, 64, 180, 2, 76, 2, 74, 2, 73, 52, 2, 71, 180, 2, 70, 54, 52, 2, 79, 52, 2, 78, 50, 2, 76, 178, 2, 74, 178, 2, 73, 52, 2, 71, 50, 2, 70, 49, 2, 71, 47, 2, 73, 45, 2, 71, 43, 2, 73, 42, 2, 74, 40, 2, 74, 73, 38, 2, 74, 73, 40, 2, 73, 71, 42, 2, 74, 73, 35, 2, 73, 163, 2, 71, 47, 2, 73, 46, 2, 74, 47, 2, 76, 49, 2, 78, 50, 2, 206, 42, 2, 79, 78, 43, 2, 78, 45, 2, 76, 43, 2, 204, 40, 2, 52, 2, 50, 2, 76, 49, 2, 204, 47, 2, 78, 76, 46, 2, 76, 55, 2, 74, 54, 2, 73, 52, 2, 78, 50, 2, 76, 49, 2, 79, 47, 2, 78, 46, 2, 78, 47, 2, 206, 42, 2, 206, 43, 2, 79, 40, 2, 74, 73, 42, 2, 74, 73, 73, 170, 2, 74, 73, 42, 2, 71, 170, 2, 71, 35, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 199, 163, 2, 1]
list_to_midi(test_events, 120, '/Users/mac/Desktop/test', 'test_list_to_midi')


# list pkl to midi
test_list_path = '/Users/mac/Desktop/test/poly_S_E_to_list.pkl'
with open(test_list_path, 'rb') as tl_file:
    test_list = pkl.load(tl_file)
    tl_file.close()

for i in range(10):
    list_to_midi(test_list[i], 120, '/Users/mac/Desktop/test', 'test_pkl_to_midi%d'%i)

'''

'''

~/sss/AAAI/common/Mag_Data/midi_tf/notesequences_midishare_Wikifonia.tfrecord

'''