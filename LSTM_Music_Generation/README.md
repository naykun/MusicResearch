# LSTM Music Generation

## 目录位置

~/sss/MusicResearch



## 运行指令

python3 train.py --layer_size=64 \
--notes_range=38 \
--batch_size=256 
--epochs=5 \

--window_size = 5\

--sequence_example_dir=Wikifonia_basic_rnn_sequence_examples/eval_melodies.tfrecord



## 实验目标

TimeStep:[1,4,8,16,32]
Embedding Length:[1,4,8,16,32]
用一样的数据集（小一点的），每次只变一个参数。
关注训练Accuracy，Loss。每个模型下最高准确率。最后有时间导出一下结果对比。
测试集Accuracy， Loss，每个模型下最高测试准确率



## 现在

在我的代码里，TimeStep应该是layer_size, Embedding Length对应window_size，我现在还不会改，但预留了一个位置

不过我把log，Tensorboard等等加好了，optimizer换成了text一样的，也加入了learning_rate。

mnist_tfrecord中训练使用了tensor，但是最后evaluate的时候并不是，这个有点坑，我找了很久没有发现在evaluate的时候直接用tensor的方法，我只能想到把模型整个设成untrainable的，然后在测试集上跑一个acc和loss出来。



sh server_dataset.sh
即可启动多卡实验用于参数探索