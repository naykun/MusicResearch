import os
#Finished: Bach 5
#TODO
#Wiki 5,6,7
if __name__ == '__main__':
    rlaunch = 'rlaunch --cpu=2 --memory=20000 --gpu=1 --preemptible=no '
    for id in [6]:
        os.system(rlaunch + 'python3 music_text_generator_conv.py %d' % (id))

