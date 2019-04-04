import pdb
import numpy as np
import keras
from keras.models import Model, load_model
from keras.layers import GRU, Input, LSTM, Bidirectional, Embedding, Dense, BatchNormalization
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
import pickle as pkl
import string
import re
from textacy.preprocess import preprocess_text
from keras.preprocessing.text import text_to_word_sequence
import os
import dill as dpickle


class Seq2seq():


    def __init__(self, latent_dim, encoder_len, decoder_len):

        self.latent_dim = latent_dim
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.body_pp = dpickle.load(open('data/body_pp.dpkl', 'rb'))
        self.title_pp = dpickle.load(open('data/title_pp.dpkl', 'rb'))
        self.source_vocab_size = max(self.body_pp.id2token.keys())+1
        self.target_vocab_size = max(self.title_pp.id2token.keys())+1
        body_vecs = np.load('data/body_vecs.npy')
        title_vecs = np.load('data/title_vecs.npy')
        self.encoder_vecs = body_vecs
        self.decoder_vecs = title_vecs[:, :-1]
        self.output_vecs = title_vecs[:, 1:]

    
    def create_model(self):

        encoder_inputs = Input(shape=(None, ), name='Encoder-Input')
        encoder_embed = Embedding(self.source_vocab_size, self.latent_dim, name='Encoder-Embed')(encoder_inputs)
        encoder_norm = BatchNormalization(name='Encoder-Batchnorm')(encoder_embed)
        encoder_lstm = LSTM(self.latent_dim, return_state=True, name='Encoder-LSTM')
        encoder_output, state_h, state_c = encoder_lstm(encoder_norm)

        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, ), name='Decoder-Input')
        decoder_embed_func = Embedding(self.target_vocab_size, self.latent_dim, name='Decoder-Embed')
        decoder_embed = decoder_embed_func(decoder_inputs)
        decoder_norm = BatchNormalization(name='Decoder-Batchnorm')(decoder_embed)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='Decoder-LSTM')
        decoder_lstm_output, _, _ = decoder_lstm(decoder_norm, initial_state=encoder_states)
        decoder_norm2 = BatchNormalization(name='Decoder-Batchnorm2')(decoder_lstm_output)
        decoder_dense = Dense(self.target_vocab_size, activation='softmax', name='Decoder-Dense')
        decoder_outputs = decoder_dense(decoder_norm2)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        print(self.model.summary())


    def train(self, batch_size, epochs, path):

        if not os.path.exists(path+"/weights"):
            os.makedirs(path+"/weights")
        checkpoint = ModelCheckpoint(path+'/weights/{epoch:02d}.hdf5', verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=path+'/tb/', write_graph=True)
        self.model.fit([self.encoder_vecs, self.decoder_vecs], np.expand_dims(self.output_vecs, -1), batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint, tensorboard])
        self.model.save(path+'/model.h5')
        

    def decode_sequence(self, input_seq):

        input_vec = self.body_pp.transform([input_seq])
        states_value = self.encoder_model.predict(input_vec)

        target_sequence = np.array(self.title_pp.token2id['_start_']).reshape(1, 1)

        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_token, h, c = self.decoder_model.predict([target_sequence] + states_value)

            pred_idx = np.argmax(output_token[0, -1, 2:]) + 2
            sampled_token = self.title_pp.id2token[pred_idx]

            if(sampled_token == '_end_' or len(decoded_sentence) > self.decoder_len):
                stop_condition = True
                break
            decoded_sentence.append(sampled_token)

            target_sequence = np.zeros((1,1))
            target_sequence[0, 0] = pred_idx
            states_value = [h, c]

        return ' '.join(decoded_sentence)


    def load_models(self, path):

        self.model = load_model(path+'/model.h5')

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
        decoder_norm1 = self.model.get_layer('Decoder-Batchnorm')
        decoder_bn = decoder_norm1(decoder_embed)
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_bn, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_norm2 = self.model.get_layer('Decoder-Batchnorm2')
        decoder_dense = self.model.get_layer('Decoder-Dense')
        decoder_bn2 = decoder_norm2(decoder_outputs)
        decoder_outputs = decoder_dense(decoder_bn2)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    
if __name__ == '__main__':

    path = 'ckpt'
    s = Seq2seq(300, 70, 12)
    s.create_model()

    s.load_models(path)

    while True:
        input_seq = input('Type Text:\n')
        decoded_sentence = s.decode_sequence(input_seq)
        print('Decoded sentence:', decoded_sentence)
    pdb.set_trace()
