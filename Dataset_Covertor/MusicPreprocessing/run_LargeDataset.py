import os
import time
import gc

start_time = time.time()

if __name__ == '__main__':
    start_time = time.time()
    for i in range(9, 32):
        cmd = 'python3 MusicAnalyse_TimeStep_LargeDataset.py %d ' % (i)
        os.system(cmd)
        print('In length %d Final Time cost:' % i, time.time() - start_time)
        gc.collect()

    for i in range(40, 100, 10):
        cmd = 'python3 MusicAnalyse_TimeStep_LargeDataset.py %d ' % (i)
        os.system(cmd)
        print('In length %d Final Time cost:' % i, time.time() - start_time)
        gc.collect()
