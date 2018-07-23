python3 basic_rnn_np_output_to_midi.py --config=basic_rnn \
--run_dir=logdir/run1 \
--output_dir=generated \
--num_outputs=1 \
--num_steps=10 \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--primer_melody="[60,60,67,67]"
