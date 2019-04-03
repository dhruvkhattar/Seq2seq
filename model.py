import pdb
import numpy as np
import keras
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Bidirectional, Embedding, Dense, BatchNormalization
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
import pickle as pkl
import string
import re
from textacy.preprocess import preprocess_text
from keras.preprocessing.text import text_to_word_sequence


class Seq2seq():


    def __init__(self, latent_dim, encoder_len, decoder_len):

        self.latent_dim = latent_dim
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.source_vocab = pkl.load(open('data/source_vocab.pkl', 'rb'))
        self.target_vocab = pkl.load(open('data/target_vocab.pkl', 'rb'))
        self.reverse_target_vocab = dict([[v,k] for k,v in self.target_vocab.items()])
        self.reverse_source_vocab = dict([[v,k] for k,v in self.source_vocab.items()])
        self.source_vocab_size = len(self.source_vocab)+1
        self.target_vocab_size = len(self.target_vocab)+1
        self.encoder_vecs = np.load('data/encoder_vecs.npy')
        self.decoder_vecs = np.load('data/decoder_vecs.npy')
        self.output_vecs = np.load('data/output_vecs.npy')

    
    def process_text(self, text):

        return text_to_word_sequence(preprocess_text(text,
                                                    fix_unicode=True,
                                                    lowercase=True,
                                                    no_urls=True,
                                                    no_emails=True,
                                                    no_phone_numbers=True,
                                                    no_numbers=True,
                                                    no_currency_symbols=True,
                                                    no_punct=True,
                                                    no_contractions=False,
                                                    no_accents=True))


    def create_model(self):

        encoder_inputs = Input(shape=(None, ), name='Encoder-Input')
        encoder_embed = Embedding(self.source_vocab_size, self.latent_dim, mask_zero=True, name='Encoder-Embed')(encoder_inputs)
        encoder_lstm = LSTM(self.latent_dim, return_state=True, name='Encoder-LSTM')
        encoder_output, state_h, state_c = encoder_lstm(encoder_embed)

        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, ), name='Decoder-Input')
        decoder_embed_func = Embedding(self.target_vocab_size, self.latent_dim, mask_zero=True, name='Decoder-Embed')
        decoder_embed = decoder_embed_func(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='Decoder-LSTM')
        decoder_lstm_output, _, _ = decoder_lstm(decoder_embed, initial_state=encoder_states)
        decoder_dense = Dense(self.target_vocab_size, activation='softmax', name='Decoder-Dense')
        decoder_outputs = decoder_dense(decoder_lstm_output)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        print(self.model.summary())


    def train(self, batch_size, epochs):

        checkpoint = ModelCheckpoint('ckpt/weights/{epoch:02d}.hdf5', verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir='ckpt/tb/', write_graph=True)
        self.model.fit([self.encoder_vecs, self.decoder_vecs], np.expand_dims(self.output_vecs, -1), batch_size=batch_size, epochs=epochs, validation_split=0.05, callbacks=[checkpoint, tensorboard])
        self.model.save('ckpt/model.h5')
 
 
    def decode_sequence(self, input_seq):

        tokens = self.process_text(input_seq)[:self.encoder_len]
        seq_len = len(tokens)
        input_vec = []
        for i in range(self.encoder_len - seq_len):
            input_vec.append(0)
        for token in tokens:
            if token in self.source_vocab:
                input_vec.append(self.source_vocab[token])
            else:
                input_vec.append(self.source_vocab['OOV'])
	
        print(input_vec)
        states_value = self.encoder_model.predict(input_vec)

        target_sequence = np.zeros((1,1))
        target_sequence[0, 0] = self.target_vocab['<s>']

        stop_condition = False
        decoded_sentence = ['<s>']
        while not stop_condition:
            output_token, h, c = self.decoder_model.predict([target_sequence] + states_value)

            pred_idx = np.argmax(output_token[0, -1, 2:]) + 2
            sampled_token = self.reverse_target_vocab[pred_idx]

            decoded_sentence.append(sampled_token)
            if(sampled_token == '</s>' or len(decoded_sentence) > self.decoder_len):
                stop_condition = True

            target_sequence = np.zeros((1,1))
            target_sequence[0, 0] = pred_idx
            states_value = [h, c]

        return ' '.join(decoded_sentence)


    def load_models(self, path):

        self.model = load_model(path)

        encoder_inputs = self.model.get_layer('Encoder-Input').input
        encoder_outputs, state_h_enc, state_c_enc = self.model.get_layer('Encoder-LSTM').output
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = self.model.get_layer('Decoder-Input').input
        decoder_state_input_h = Input(shape=(self.latent_dim, ))
        decoder_state_input_c = Input(shape=(self.latent_dim, ))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = self.model.get_layer('Decoder-LSTM')
        decoder_embed_func = self.model.get_layer('Decoder-Embed')
        decoder_embed = decoder_embed_func(decoder_inputs)
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_embed, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = self.model.get_layer('Decoder-Dense')
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


if __name__ == '__main__':

    s = Seq2seq(300, 70, 12)
    s.create_model()
    s.train(1024, 10)

    s.load_models('ckpt/model.h5')

    while True:
        input_seq = input('Type Text:\n')
        decoded_sentence = s.decode_sequence(input_seq)
        print('Decoded sentence:', decoded_sentence)
    pdb.set_trace()
