python3 main.py --layer_size=128 \
    --notes_range=38 \
    --batch_size=256 \
    --predict_batch_size=1 \
    --Ty=100 \
    --epochs=100 \
    --embedding_len=8 \
    --sequence_example_train_file=/home/ouyangzhihao/sss/AAAI/yyh/Wikifonia_basic_rnn_sequence_examples/training_melodies.tfrecord \
    --sequence_example_eval_file=/home/ouyangzhihao/sss/AAAI/yyh/Wikifonia_basic_rnn_sequence_examples/eval_melodies.tfrecord \
    --maxlen=200 \
    --encoding_config=basic_rnn \
    --output_dir=generated \
    --num_outputs=5 \
    --num_steps=10 \
    --primer_melody="[60,-2,60,-2,67,-2,67,-2]"

