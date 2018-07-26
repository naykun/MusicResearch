# import os
#
# how_much_part = 10
# for how_much_part in [1,10,100,10000]:
#     cmd = 'python3 tmp.py %d %d %d' % (512, 1, how_much_part)
#     os.system(cmd)

#1-20
cmd = []
layer_size = 128
window_size = 20

import os

try:
	os.mkdir("cmd")
except:
	pass

layer_size_list = [1,4,8,16,32,64]
window_size_list = [1,4,8,16,32,64]

for layer_size in layer_size_list:
	for window_size in window_size_list:
		print( '''python3 train.py --epochs=200 \\
			--sequence_example_dir=Wikifonia_basic_rnn_sequence_examples/eval_melodies.tfrecord \\
			--notes_range=38 \\
			--batch_size=2048 \\
			--layer_size=%d \\
			--window_size=%d'''% (layer_size, window_size) , 
			file = open("cmd/cmd_LayerSize%d_WindowSize%d.sh"%(layer_size, window_size),"w"))

if __name__ == '__main__':
    rlaunch = 'rlaunch --preemptible=no --cpu=4 --gpu=1 --memory=4096 '
    for layer_size in layer_size_list:
    	for window_size in window_size_list:
        	cmd.append(rlaunch + "sh cmd/cmd_LayerSize%d_WindowSize%d.sh"%(layer_size, window_size))
    # print(cmd)

    print('\n'.join(cmd))
