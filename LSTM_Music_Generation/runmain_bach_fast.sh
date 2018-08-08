python3 main.py --layer_size=128 \
    --notes_range=38 \
    --batch_size=256 \
    --predict_batch_size=1 \
    --Ty=32 \
    --epochs=500 \
    --embedding_len=2 \
    --sequence_example_train_file=/home/ouyangzhihao/sss/AAAI/common/Mag_Data/midi_S_E/S_E_BasicRNN_midishare_bach20180725/training_melodies.tfrecord \
    --sequence_example_eval_file=/home/ouyangzhihao/sss/AAAI/common/Mag_Data/midi_S_E/S_E_BasicRNN_midishare_bach20180725/eval_melodies.tfrecord \
    --maxlen=80 \
    --encoding_config=basic_rnn \
    --output_dir=generated \
    --num_outputs=5 \
    --num_steps=10 \
    --primer_melody="[60,-2,60,-2,67,-2,67,-2]"

