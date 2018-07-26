python3 melody_rnn_generate.py --config=basic_rnn \
--run_dir=logdir/run1 \
--output_dir=generated \
--num_outputs=1 \
--num_steps=10 \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--primer_melody="[60]"
