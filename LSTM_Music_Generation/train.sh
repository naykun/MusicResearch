python3 melody_rnn_train.py --config=basic_rnn \
--run_dir=logdir/run1 \
--sequence_example_file=Wikifonia_basic_rnn_sequence_examples/eval_melodies.tfrecord \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--num_training_steps=1
