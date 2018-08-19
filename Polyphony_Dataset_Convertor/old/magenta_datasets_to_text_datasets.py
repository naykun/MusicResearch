import numpy as np
import tensorflow as tf
import keras
import os
import pickle
from my_to_midi import *
from sequence_example_lib import *
import copy

def to_events(x):
    return np.argmax(x, axis=1)
def to_real_length(x):
    while(x[len(x)-1]==0):
        x.pop()
    # delete the last note along with padded zeros
    x.pop()

index_to_char_str = "._0123456789abcdefghijklmnopqrstuvwxyz"
char_to_index = dict((c, i) for i, c in enumerate(index_to_char_str))
index_to_char = dict((i, c) for i, c in enumerate(index_to_char_str))

# magenta dataset
def sequence_example_to_text(sequence_example_path, output_dir, name):
    # train
    sequence_example_file = sequence_example_path
    sequence_example_file_paths = tf.gfile.Glob(
        os.path.expanduser(sequence_example_file))
    start_time = time.time()
    inputs, labels, lengths = get_numpy_from_tf_sequence_example( input_size=38,
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
    output_path = os.path.join(output_dir, name+'.txt')
    with open(output_path, "w") as wf:
        for i in real_inputs:
            for j in range(len(i)):
                wf.write(index_to_char_str[i[j]])
        wf.close()

def magenta_sequence_example_to_text(sequence_example_dir, output_dir, name):
    train_path = os.path.join(sequence_example_dir, "training_melodies.tfrecord")
    sequence_example_to_text(train_path, output_dir, name+"_train")
    eval_path = os.path.join(sequence_example_dir, "eval_melodies.tfrecord")
    sequence_example_to_text(eval_path, output_dir, name+"_eval")


magenta_datasets_dirs = [
    '/Users/mac/Desktop/MusicGeneration/Mag_Data/S_E_BasicRNN_midishare_bach20180725',
    '/Users/mac/Desktop/MusicGeneration/Mag_Data/S_E_BasicRNN_midishare_Beethoven_20180725',
    '/Users/mac/Desktop/MusicGeneration/Mag_Data/S_E_BasicRNN_midishare_Mozart20180725',
    '/Users/mac/Desktop/MusicGeneration/Mag_Data/S_E_BasicRNN_midishare_Wikifonia20180725'
    ]
magenta_datasets_names = ['Bach',
                          'Beethoven',
                          'Mozart',
                          'Wikifonia']
for dataset_dir, dataset_name in zip(magenta_datasets_dirs, magenta_datasets_names):
    print("Converting:", dataset_dir, dataset_name)
    magenta_sequence_example_to_text(dataset_dir, "/Users/mac/Desktop/makedataset", dataset_name)

