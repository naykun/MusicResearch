import os

if __name__ == '__main__':
    rlaunch = 'rlaunch --cpu=2 --memory=35000 --gpu=1 --preemptible=no '
    for id in [3]:
        os.system(rlaunch + 'python3 polyphony_train_conv.py %d' % (id))