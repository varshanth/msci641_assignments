import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import re


class AmazonReviewsDS:
    def __init__(self, pos_text_file, neg_text_file, ds_cfg):
        pos_reviews = neg_reviews = None
        print('----- Dataset Synthesis Start -----')

        print(f'Loading Positive Reviews from {pos_text_file}')
        with open(pos_text_file, 'r') as p_txt:
            pos_reviews = [review.lower() for review in p_txt.read().split('\n')[:-1]]
        assert pos_reviews != None, 'Positive Reviews Load Failed'

        print(f'Loading Negative Reviews from {pos_text_file}')
        with open(neg_text_file, 'r') as n_txt:
            neg_reviews = [review.lower() for review in n_txt.read().split('\n')[:-1]]
        assert neg_reviews != None, 'Negative Reviews Load Failed'
        self.ds_cfg = ds_cfg

        print('Generating data and labels')
        self.data = np.array(pos_reviews + neg_reviews)
        self.labels = np.append(np.ones(len(pos_reviews)), np.zeros(len(neg_reviews)))

        # Preprocessing
        self._tokenize()
        if ds_cfg['stopword_removal']:
            self._remove_stop_words()
        self._shuffle_data()

        print('----- Dataset Synthesis Complete -----')

    def _tokenize(self):
        print('Tokenizing the data')
        self.data = np.array(list(map(
            lambda review: re.findall('[a-zA-Z]+|[0-9]+|[\.\'\-_,]+', review), self.data)))

    def _remove_stop_words(self):
        print('Removing stop words')
        eng_stopwords = set(stopwords.words('english'))
        self.data = np.array(list(
            list(filter(lambda token: token not in eng_stopwords, review))
            for review in self.data))

    def _shuffle_data(self):
        print('Shuffling the data')
        shuffle_idx = np.random.permutation(len(self.data))
        self.data = self.data[shuffle_idx]
        self.labels = self.labels[shuffle_idx]

    def __len__(self):
        return len(self.data)

    def get_train_data(self):
        total = sum(self.ds_cfg['splits'].values())
        train_idx = (len(self) * self.ds_cfg['splits']['train']) // total
        return (self.data[:train_idx], self.labels[:train_idx])

    def get_val_data(self):
        total = sum(self.ds_cfg['splits'].values())
        train_idx = (len(self) * self.ds_cfg['splits']['train']) // total
        val_idx = train_idx + ((len(self) * self.ds_cfg['splits']['val']) // total)
        return (self.data[train_idx:val_idx], self.labels[train_idx:val_idx])

    def get_test_data(self):
        total = sum(self.ds_cfg['splits'].values())
        test_idx = (len(self) * self.ds_cfg['splits']['test']) // total
        return (self.data[-test_idx:], self.labels[-test_idx:])

