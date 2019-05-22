from lib.amazon_reviews_loader import AmazonReviewsDS
from lib.amazon_reviews_cfg import DS_CFG_NO_SW, DS_CFG_SW


_POS_REV_FILE = 'dataset/pos.txt'
_NEG_REV_FILE = 'dataset/neg.txt'


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
    amazon_rev_no_sw = AmazonReviewsDS(_POS_REV_FILE, _NEG_REV_FILE, DS_CFG_NO_SW)
    print('Retrieving Amazon Reviews Dataset with Stopwords')
    amazon_rev_sw = AmazonReviewsDS(_POS_REV_FILE, _NEG_REV_FILE, DS_CFG_SW)

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

    print('--------------------------------')
    print('Amazon Reviews with No Stopwords:')
    print('--------------------------------')
    print_ds_info(amazon_rev_no_sw_splits)
    print('\n')

    print('--------------------------------')
    print('Amazon Reviews with Stopwords:')
    print('--------------------------------')
    print_ds_info(amazon_rev_sw_splits)

