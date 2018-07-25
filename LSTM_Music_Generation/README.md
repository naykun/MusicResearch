# LSTM Music Generation

## Train and Evaluate

有待渣渣坤添加。



## Model Output to MIDI

### 功能

将Keras模型输出，也就是one-hot的numpy矩阵，转化为MIDI，也可以解析primer_melody，以one-hot的形式返回，作为Keras模型的输入。

### 依赖

需要magenta环境，必须是我修改的这个magenta环境，部分magenta源码进行了修改，只要所有的py文件和magenta目录在同一级，就可以运行。

### 运行方法

先运行训练，“--config”、“--hparams”、“--num_training_steps”三个参数不要改动，路径可以更换，运行训练的目的是为了生成model的bundle_file和check_point，以便生成，这个模型跟生成的结果无关，只是方便了解码，但这一步是必须的，运行命令在 **train.sh** 中：

```python
python3 melody_rnn_train.py --config=basic_rnn \
--run_dir=logdir/run1 \
--sequence_example_file=Wikifonia_basic_rnn_sequence_examples/eval_melodies.tfrecord \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--num_training_steps=1
```

然后运行 **basic_rnn_np_output_to_midi.py** ，有如下方法可以使用。

```python
def event_sequence_to_midi(generator, encoded_event_sequence, index, config):
def get_primer_events(generator, config):
def one_hot_to_encoded_event_sequence(one_hot_output):
def encoded_event_sequence_to_one_hot(encoded_event_sequence, input_size):
```

其中，所有的event_sequence都是指python-list，每个元素是一个int，代表一个event。

每个one_hot都是numpy矩阵，其中转化过程需要提供one_hot的维数，在basic_rnn中是38。

运行命令在 **my_generate.sh** 中，具体内容如下：

```python
python3 basic_rnn_np_output_to_midi.py --config=basic_rnn \
--run_dir=logdir/run1 \
--output_dir=generated \
--num_outputs=5 \
--num_steps=10 \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--primer_melody="[60,-2,60,-2,67,-2,67,-2]"
```

其中**--num_steps** 可以在程序中通过FLAGS获取，作为Keras的生成序列长度使用。

另外两个脚本 **test.sh、clean.sh** 暂时用处不大。