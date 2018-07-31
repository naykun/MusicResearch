import os
import time
import gc
start_time = time.time()

if __name__ == '__main__':
    for i in range(1,60):
        cmd = 'python3 MusicAnalyse_TimeStep.py %d ' % (i)
        os.system(cmd)
        print('In length %d Final Time cost:' % i, time.time() - start_time)
        gc.collect()
