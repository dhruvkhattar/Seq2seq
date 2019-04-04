import pdb
import numpy as np
import pandas as pd
from collections import defaultdict
from collections import Counter
import spacy
from tqdm import tqdm
import pickle as pkl
from ktext.preprocess import processor
import dill as dpickle


class data_handler():


    def __init__(self, filename, encoder_vocab_size, decoder_vocab_size, max_encoder_len, max_decoder_len):

        print('Initiating Data Loader')
        self.filename = filename
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len


    def load_data(self):

        source_vocab_freq = defaultdict(int)
        target_vocab_freq = defaultdict(int)

        self.df = pd.read_csv(self.filename).sample(1000000)
        self.body = self.df['body'].tolist()
        self.titles = self.df['issue_title'].tolist()

        body_pp = processor(keep_n=self.encoder_vocab_size, padding_maxlen=self.max_encoder_len)
        body_vecs = body_pp.fit_transform(self.body)

        title_pp = processor(append_indicators=True, keep_n=self.decoder_vocab_size, padding_maxlen=self.max_decoder_len, padding='post')
        title_vecs = title_pp.fit_transform(self.titles)

        dpickle.dump(body_pp, open('data/body_pp.dpkl', 'wb'))
        dpickle.dump(title_pp, open('data/title_pp.dpkl', 'wb'))

        np.save('data/body_vecs.npy', body_vecs)
        np.save('data/title_vecs.npy', title_vecs)
        
if __name__ == '__main__':

    d = data_handler('data/github_issues.csv', 8000, 4500, 70, 12)
    d.load_data()
