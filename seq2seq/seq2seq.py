from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64  
epochs = 100  
latent_dim = 256 
num_samples = 10000  
data_path = 'Bach.txt'



# prepare datasets
# -------------------------------------------------


input_melodies = []
target_melodies = []
input_notes = set()
target_notes = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    
    #reverse as target
    input_melody = line
    target_melody = reversed(input_melody)


    target_melody = '\t' + target_melody + '\n'
    input_melodies.append(input_melody)
    target_melodies.append(target_melody)
    for char in input_melody:
        if char not in input_notes:
            input_notes.add(char)
    for char in target_melody:
        if char not in target_notes:
            target_notes.add(char)

input_notes = sorted(list(input_notes))
target_notes = sorted(list(target_notes))
num_encoder_tokens = len(input_notes)
num_decoder_tokens = len(target_notes)
max_encoder_seq_length = max([len(txt) for txt in input_melodies])
max_decoder_seq_length = max([len(txt) for txt in target_melodies])

print('Number of samples:', len(input_melodies))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_notes)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_notes)])

encoder_input_data = np.zeros(
    (len(input_melodies), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_melodies), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_melodies), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_melody, target_melody) in enumerate(zip(input_melodies, target_melodies)):
    for t, char in enumerate(input_melody):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_melody):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.




#    build net work
# ------------------------------------------------------------------ 


encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c]


decoder_inputs = Input(shape=(None, num_decoder_tokens))


decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

model.save('seq2seq.h5')



encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):

    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))

    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]
    return decoded_sentence


for seq_index in range(100):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_melodies[seq_index])
    print('Decoded sentence:', decoded_sentence)
