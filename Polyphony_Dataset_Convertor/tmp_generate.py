import ast
import os
import time

# internal imports
import pickle as pkl

import tensorflow as tf
import magenta

import polyphony_lib
import polyphony_encoder_decoder
import numpy as np
from  sequence_example_lib import *
from my_to_midi import *
import pickle as pkl
from polyphony_dataset_convertor import *
import magenta.music as mm





test_list_path = '/unsullied/sharefs/ouyangzhihao/Share/LSTM/Text_Generation_Capacity/Code/Music_Research_Exp/Music_Text_Gneration/9_1_AAAI_single_multi_track/multi_track_bach_finish/logdir/resNet_Local_Bach_batchS128_epochs150_maxL256_step128_embeddingL1_2018-09-02_090538/data/9_train_diversity02.pkl'
with open(test_list_path, 'rb') as tl_file:
    test_list = pkl.load(tl_file)
    tl_file.close()

target_path = '/unsullied/sharefs/ouyangzhihao/Share/LSTM/Text_Generation_Capacity/Code/Music_Research_Exp/Music_Text_Gneration/9_1_AAAI_single_multi_track/multi_track_bach_finish/logdir/resNet_Local_Bach_batchS128_epochs150_maxL256_step128_embeddingL1_2018-09-02_090538/midi_more/8_5_superlong/'
print(np.shape(test_list))
# import ipdb
import ipdb; ipdb.set_trace()
list_to_midi(test_list[:100], 120, target_path, 'test_pkl_to_midi%d'%100)
list_to_midi(test_list[:1000], 120, target_path, 'test_pkl_to_midi%d'%1000)
list_to_midi(test_list[:10000], 120, target_path, 'test_pkl_to_midi%d'%10000)
# for i in range(10):
# list_to_midi(test_list, 120, target_path, 'test_pkl_to_midi%d'%1)
