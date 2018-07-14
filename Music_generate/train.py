import keras

import melody_rnn_create_dataset
import sequence_example_lib


def noteseq2seqexmp(noteseqfile):
"""
    TODO：解耦config、读取notesequencefile
    文件：create_dataset
"""
    pipeline_inst = melody_rnn_create_dataset.get_pipeline(self.config,
                                                           eval_ratio=0.0)
    result = pipeline_inst.transform(note_sequence)

def seqexmp2inputs(seqexmp_file_paths,mode,batch_size, input_size):
    inputs, labels, lengths = sequence_example_lib.get_padded_batch(
          seqexmp_file_paths, batch_size, input_size,
          shuffle=mode == 'train')
    return inputs,labels,lengths




