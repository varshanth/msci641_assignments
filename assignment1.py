from sys import argv
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import re
from copy import copy

DS_CFG_NO_SW = {
        'splits' : {
            'train' : 80,
            'val' : 10,
            'test' : 10
            },
        'stopword_removal' : True
        }

DS_CFG_SW = copy(DS_CFG_NO_SW)
DS_CFG_SW['stopword_removal'] = False

class AmazonReviewsDS:
    def __init__(self, text_file, ds_cfg):
        reviews = None
        print('----- Dataset Synthesis Start -----')

        print(f'Loading Reviews from {text_file}')
        with open(text_file, 'r') as txt:
            reviews = txt.read().split('\n')[:-1]
            reviews = list(map(lambda review: review.lower(), reviews))
        assert reviews != None, 'Reviews Load Failed'
        self.ds_cfg = ds_cfg
        self.data = reviews

        # Preprocessing
        self._tokenize()
        if ds_cfg['stopword_removal']:
            self._remove_stop_words()
        self._shuffle_data()

        print('----- Dataset Synthesis Complete -----')

    def _tokenize(self):
        print('Tokenizing the data')
        self.data = list(map(lambda review: re.sub('[^\w\-\'\.,\s]', '', review), self.data))
        self.data = list(map(
            lambda review: re.findall('[a-zA-Z]+|[0-9]+|[\.\'\-_,]+', review), self.data))

    def _remove_stop_words(self):
        print('Removing stop words')
        eng_stopwords = set(stopwords.words('english'))
        self.data = list(list(filter(lambda token: token not in eng_stopwords, review))
            for review in self.data)

    def _shuffle_data(self):
        print('Shuffling the data')
        shuffle_idx = np.random.permutation(len(self.data))
        self.data = list(np.array(self.data)[shuffle_idx])

    def __len__(self):
        return len(self.data)

    def get_train_data(self):
        total = sum(self.ds_cfg['splits'].values())
        train_idx = (len(self) * self.ds_cfg['splits']['train']) // total
        return self.data[:train_idx]

    def get_val_data(self):
        total = sum(self.ds_cfg['splits'].values())
        train_idx = (len(self) * self.ds_cfg['splits']['train']) // total
        val_idx = train_idx + ((len(self) * self.ds_cfg['splits']['val']) // total)
        return self.data[train_idx:val_idx]

    def get_test_data(self):
        total = sum(self.ds_cfg['splits'].values())
        test_idx = (len(self) * self.ds_cfg['splits']['test']) // total
        return self.data[-test_idx:]

def print_ds_info(splits):
    def print_first_n_data(split, n):
        print(f'First {n} Reviews: {split[0][:n]}')
        print(f'First {n} Labels: {split[1][:n]}')

    print(f'Len(Train): {len(splits["train"][0])}')
    print_first_n_data(splits["train"], 3)
    print(f'Len(Val): {len(splits["val"][0])}')
    print_first_n_data(splits["val"], 3)
    print(f'Len(Test): {len(splits["test"][0])}')
    print_first_n_data(splits["test"], 3)

if __name__ ==  '__main__':
    print('Retrieving Amazon Reviews Dataset with No Stopwords')
    amazon_rev_no_sw = AmazonReviewsDS(argv[1], DS_CFG_NO_SW)
    print('Retrieving Amazon Reviews Dataset with Stopwords')
    amazon_rev_sw = AmazonReviewsDS(argv[1], DS_CFG_SW)

    print('Splitting the datasets into train, validation and test sets')
    amazon_rev_no_sw_splits = {
            'train' : amazon_rev_no_sw.get_train_data(),
            'val': amazon_rev_no_sw.get_val_data(),
            'test' : amazon_rev_no_sw.get_test_data()
            }

    amazon_rev_sw_splits = {
            'train' : amazon_rev_sw.get_train_data(),
            'val': amazon_rev_sw.get_val_data(),
            'test' : amazon_rev_sw.get_test_data()
            }

    print('Saving the splits into files')
    np.savetxt("train.csv", amazon_rev_sw_splits['train'], delimiter=",", fmt='%s')
    np.savetxt("val.csv", amazon_rev_sw_splits['val'], delimiter=",", fmt='%s')
    np.savetxt("test.csv", amazon_rev_sw_splits['test'], delimiter=",", fmt='%s')

    np.savetxt("train_no_stopword.csv", amazon_rev_no_sw_splits['train'],
            delimiter=",", fmt='%s')
    np.savetxt("val_no_stopword.csv", amazon_rev_no_sw_splits['val'],
            delimiter=",", fmt='%s')
    np.savetxt("test_no_stopword.csv", amazon_rev_no_sw_splits['test'],
        delimiter=",", fmt='%s')
    print('Done!')

