from assignment1 import AmazonReviewsDS, DS_CFG_NO_SW, DS_CFG_SW

_POS_TXT = 'dataset/pos.txt'
_NEG_TXT = 'dataset/neg.txt'

if __name__ ==  '__main__':
    amazon_rev_no_sw_pos = AmazonReviewsDS(_POS_TXT, DS_CFG_NO_SW)
    amazon_rev_sw_pos = AmazonReviewsDS(_POS_TXT, DS_CFG_SW)
    amazon_rev_no_sw_neg = AmazonReviewsDS(_NEG_TXT, DS_CFG_NO_SW)
    amazon_rev_sw_neg = AmazonReviewsDS(_NEG_TXT, DS_CFG_SW)

    print('Splitting the datasets into train, validation and test sets')
    amazon_rev_no_sw_pos_splits = {
            'train' : amazon_rev_no_sw_pos.get_train_data(),
            'val': amazon_rev_no_sw_pos.get_val_data(),
            'test' : amazon_rev_no_sw_pos.get_test_data()
            }
    amazon_rev_sw_pos_splits = {
            'train' : amazon_rev_sw_pos.get_train_data(),
            'val': amazon_rev_sw_pos.get_val_data(),
            'test' : amazon_rev_sw_pos.get_test_data()
            }
    amazon_rev_no_sw_neg_splits = {
            'train' : amazon_rev_no_sw_neg.get_train_data(),
            'val': amazon_rev_no_sw_neg.get_val_data(),
            'test' : amazon_rev_no_sw_neg.get_test_data()
            }
    amazon_rev_sw_neg_splits = {
            'train' : amazon_rev_sw_neg.get_train_data(),
            'val': amazon_rev_sw_neg.get_val_data(),
            'test' : amazon_rev_sw_neg.get_test_data()
            }






