import pdb
import numpy as np
import pandas as pd
from collections import defaultdict
from collections import Counter
import spacy
from tqdm import tqdm
import pickle as pkl
from spacy.lang.en import English
from textacy.preprocess import preprocess_text
import keras
from keras.preprocessing.text import text_to_word_sequence


class data_handler():


    def __init__(self, filename, encoder_vocab_size, decoder_vocab_size, max_encoder_len, max_decoder_len):

        print('Initiating Data Loader')
        self.filename = filename
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len


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

    
    def load_data(self):

        source_vocab_freq = defaultdict(int)
        target_vocab_freq = defaultdict(int)

        self.df = pd.read_csv(self.filename).sample(2000000)
        self.body = self.df['body'].tolist()
        self.titles = self.df['issue_title'].tolist()

        for i in tqdm(range(len(self.body))):
            source_text = self.process_text(self.body[i])
            target_text = ['<s>'] + self.process_text(self.titles[i]) + ['</s>']
            
            for token in source_text:
                source_vocab_freq[token] += 1
            
            for token in target_text:
                target_vocab_freq[token] += 1

        vocab_sorted = Counter(source_vocab_freq)
        vocab_sorted = vocab_sorted.most_common(self.encoder_vocab_size-2)
        self.source_vocab = dict({v[0]:k+2 for k, v in enumerate(vocab_sorted)})
        self.source_vocab['OOV'] = 1

        vocab_sorted = Counter(target_vocab_freq)
        vocab_sorted = vocab_sorted.most_common(self.decoder_vocab_size-2)
        self.target_vocab = dict({v[0]:k+2 for k, v in enumerate(vocab_sorted)})
        self.target_vocab['OOV'] = 1

        pkl.dump(self.source_vocab, open('data/source_vocab.pkl', 'wb'))
        pkl.dump(self.target_vocab, open('data/target_vocab.pkl', 'wb'))

        encoder_vecs = []
        decoder_vecs = []
        output_vecs = []
        for i in tqdm(range(len(self.body))):
            source_text = self.process_text(self.body[i])
            target_text = ['<s>'] + self.process_text(self.titles[i]) + ['</s>']
                
            vec = []
            tokens = source_text[:self.max_encoder_len]
            seq_len = len(tokens)
            for i in range(self.max_encoder_len - seq_len):
                vec.append(0)
            for token in tokens:
                if token in self.source_vocab:
                    vec.append(self.source_vocab[token])
                else:
                    vec.append(self.source_vocab['OOV'])
            encoder_vecs.append(vec)
            
            vec = []
            vec2 = []
            tokens = target_text[:self.max_decoder_len]
            t = 0
            for token in tokens:
                if token in self.target_vocab:
                    vec.append(self.target_vocab[token])
                    if t > 0:
                        vec2.append(self.target_vocab[token])
                else:
                    vec.append(self.target_vocab['OOV'])
                    if t > 0:
                        vec2.append(self.target_vocab['OOV'])
                t += 1
            for i in range(self.max_decoder_len - len(vec)):
                vec.append(0)
            for i in range(self.max_decoder_len - len(vec2)):
                vec2.append(0)
            
            decoder_vecs.append(vec)
            output_vecs.append(vec2)

        encoder_vecs = np.array(encoder_vecs)
        decoder_vecs = np.array(decoder_vecs)
        output_vecs = np.array(output_vecs)

        np.save('data/encoder_vecs.npy', encoder_vecs)
        np.save('data/decoder_vecs.npy', decoder_vecs)
        np.save('data/output_vecs.npy', output_vecs)


if __name__ == '__main__':

    d = data_handler('data/github_issues.csv', 10000, 7000, 70, 12)
    d.load_data()
