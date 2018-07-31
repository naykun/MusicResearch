import os

maxlen = 48
dense_size = 0
embedding_length = 1
dataset_name = 'Bach'
dataset_dir = 'datasets/Bach/'

cmd = '''python3 music_text_generator.py --batch_size=1024 \
   --epochs=200 \
   --units=512 \
   --maxlen=%d \
   --generate_length=400 \
   --dense_size=%d \
   --step=1 \
   --embedding_length=%d \
   --dataset_name=%s \
   --dataset_dir=%s'''



for maxlen in [1, 4, 8, 16, 32, 64, 128]:
    # print(cmd % (maxlen, dense_size, embedding_length, dataset_name, dataset_path))
    os.system(cmd % (maxlen, dense_size, embedding_length, dataset_name, dataset_dir))

dense_size = 16
for embedding_length in [8, 16]:
    maxlen = 32
    os.system(cmd % (maxlen, dense_size, embedding_length, dataset_name, dataset_dir))

for dense_size in [16, 64]:
    maxlen = 16
    os.system(cmd % (maxlen, dense_size, embedding_length, dataset_name, dataset_dir))

# Dataset
# dataset_name = 'Wikifonia'
# dataset_path = 'datasets/Wikifonia_train.txt'



'''

python3 music_text_generator.py --batch_size=1024 \
    --epochs=10 \
    --units=128 \
    --maxlen=48 \
    --dense_size=3 \
    --step=8 \
    --embedding_length=4 \
    --dataset_name=Bach_train \
    --dataset_path=datasets/Bach_train.txt



rlanuch --cpu=4 -- python3 --

'''