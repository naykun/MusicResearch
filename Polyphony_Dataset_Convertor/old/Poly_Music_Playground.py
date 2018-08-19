import numpy as np
import tensorflow as tf
import keras
import os
import pickle
from my_to_midi import *
from sequence_example_lib import *
from math import *
import copy
import pickle as pkl


# In[ ]:


def to_events(x):
    return np.argmax(x, axis=1)
def to_real_length(x):
    while(x[len(x)-1]==0):
        x.pop()
    # delete the last note along with padded zeros
    x.pop()


# In[ ]:


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


# In[4]:


def get_start(events):
    ret = []
    for i, event in enumerate(events):
        if event != 2:
            ret = events[i:len(events)]
            break
    for i in range(len(ret)-1, -1, -1):
        if ret[i] != 2 and ret[i] != 1:
            ret = ret[0:i+1]
            break
    return ret


def get_melody(events):
    ret = []
    for i, event in enumerate(events):
        if event==2:
            pass
            #ret.append(event)
        else:
            if i>0 and events[i-1]==2:
                ret.append([event])
    return ret
def get_accomp(events):
    ret = []
    for i, event in enumerate(events):
        if event==2:
            pass
            #ret.append(event)
        else:
            if i>0 and events[i-1]==2:
                for j in range(i+1, len(events)):
                    if events[j]==2 or j==len(events)-1:
                        ret.append(events[i+1:j])
                        break
    return ret


# In[5]:


def sequence_example_to_events_file(sequence_example_path, output_dir, name):
    real_inputs = sequence_example_to_real_inputs(sequence_example_path)
    
    melodys = []
    accomps = []
    
    for i, real_input in enumerate(real_inputs):
        events = get_start(real_input)
        melody = get_melody(events)
        accomp = get_accomp(events)


        min_len = min(len(melody),len(accomp))

        melody = melody[0:min_len]
        accomp = accomp[0:min_len]

        melodys.append(melody)
        accomps.append(accomp)

        # if i==0:
        #     print(events)
        #     print(melody)
        #     print(accomp)
        # if(len(melody)!=len(accomp)):
        #     print("Not equal",len(melody),len(accomp))

    
    melody_path = os.path.join(output_dir, name)+'_melody.pkl'
    with open(melody_path,'wb') as mf:   #pickle只能以二进制格式存储数据到文件
        mf.write(pickle.dumps(melodys))   #dumps序列化源数据后写入文件
        mf.close()

    accomp_path = os.path.join(output_dir, name)+'_accomp.pkl'
    with open(accomp_path,'wb') as af:   #pickle只能以二进制格式存储数据到文件
        af.write(pickle.dumps(accomps))   #dumps序列化源数据后写入文件
        af.close()


# In[ ]:


def to_melody_and_accompaniment(sequence_example_dir, output_dir, name):
    train_path = os.path.join(sequence_example_dir, "training_poly_tracks.tfrecord")
    sequence_example_to_events_file(train_path, output_dir, name+"_train")
    eval_path = os.path.join(sequence_example_dir, "eval_poly_tracks.tfrecord")
    sequence_example_to_events_file(eval_path, output_dir, name+"_eval")


# In[ ]:


magenta_datasets_dirs = [
    '~/sss/AAAI/common/Mag_Data/poly_midi_S_E/Bach'
    ]
magenta_datasets_names = [
    'Bach'
    ]
for dataset_dir, dataset_name in zip(magenta_datasets_dirs, magenta_datasets_names):
    print("Converting:", dataset_dir, dataset_name)
    to_melody_and_accompaniment(dataset_dir, "/home/ouyangzhihao/sss/AAAI/common/Mag_Data/Poly_List_Datasets", dataset_name)

