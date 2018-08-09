# import os
#
# how_much_part = 10
# for how_much_part in [1,10,100,10000]:
#     cmd = 'python3 tmp.py %d %d %d' % (512, 1, how_much_part)
#     os.system(cmd)

#1-20
cmd = []
how_much_part = 1
layer_size = 128
win_size = 20
if __name__ == '__main__':
    rlaunch = 'rlaunch --preemptible=no --cpu=4 --gpu=1 --memory=4096 '
    for how_much_part in [10]:
        for layer_size in [512]:
            for win_size in [1,5,10,20,30,40]:
                cmd.append(rlaunch + 'python3 text_generator.py %d %d %d' % (layer_size, win_size, how_much_part)
                   )
    # print(cmd)

    print('\n'.join(cmd))


