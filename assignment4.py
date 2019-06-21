#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gensim import models
from lib.amazon_reviews_loader import AmazonReviewsDS
from lib.amazon_reviews_cfg import DS_CFG_NO_SW, DS_CFG_SW
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, Flatten, Activation
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import GridSearchCV
from copy import copy

_POS_REV_FILE = 'dataset/pos.txt'
_NEG_REV_FILE = 'dataset/neg.txt'
_WORD2VEC_EMBEDDING = 'dataset/word2vec_embeddings.kv'
_MAX_VOCAB_SIZE = 20000
_EMBEDDING_DIM = 300


# In[ ]:


amazon_rev_sw = AmazonReviewsDS(_POS_REV_FILE, _NEG_REV_FILE, DS_CFG_SW)


# In[ ]:


_max_len_sentence = int(np.percentile([len(rev) for rev in amazon_rev_sw.data], 90))


# In[ ]:


print('Fitting Tokenizer on Dataset')
tokenizer = Tokenizer(num_words = _MAX_VOCAB_SIZE)
tokenizer.fit_on_texts([' '.join(rev[:_max_len_sentence]) for rev in amazon_rev_sw.data])
X = tokenizer.texts_to_sequences([' '.join(rev[:_max_len_sentence]) for rev in amazon_rev_sw.data])
X = pad_sequences(X, maxlen=_max_len_sentence, padding='post', truncating='post')


# In[ ]:


print('Splitting Dataset into Train, Val & Test')
X_train, X_val_test, y_train, y_val_test = train_test_split(X, amazon_rev_sw.labels, random_state=10, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, random_state=10, test_size=0.5)


# In[ ]:


print('Creating local embeddings matrix from Word2Vec Embeddings')
word2vec_embeddings = models.KeyedVectors.load(_WORD2VEC_EMBEDDING, mmap='r')
num_unique_tokens = len(tokenizer.word_index)+1  # +1 is because 0th index corresponds to pad char
embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(num_unique_tokens, _EMBEDDING_DIM))

for word, i in tokenizer.word_index.items(): # i=0 is the embedding for the zero padding
    try:
        embeddings_vector = word2vec_embeddings[word]
    except KeyError:
        embeddings_vector = None
    if embeddings_vector is not None:
        embeddings_matrix[i] = embeddings_vector

del word2vec_embeddings


# In[ ]:


print('Creating NN Model')
_NUM_HIDDEN_UNITS = 1024

def create_nn_model(activation_fn, dropout_rate, reg_param):
    print(f'Activation: {activation_fn}')
    print(f'DropoutRate: {dropout_rate}')
    print(f'RegParam: {reg_param}')
    model = Sequential()
    model.add(Embedding(input_dim = num_unique_tokens,
                        output_dim = _EMBEDDING_DIM,
                        weights = [embeddings_matrix],
                        trainable=False,
                        name='word_embedding_layer',
                        input_length = _max_len_sentence))
    # Output will be [batch size, input_length, output_dim]
    model.add(Flatten())
    model.add(Dense(_NUM_HIDDEN_UNITS,
                    activation='relu',
                    kernel_regularizer = l2(reg_param),
                    name='hidden_layer'))
    model.add(Dropout(rate=0.1, name='dropout_layer'))
    model.add(Dense(1, activation = 'sigmoid', name = 'output_layer'))
    model.summary()
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model


# In[ ]:


_N_EPOCHS = 10
_BATCH_SIZE = 32

from collections import OrderedDict
hyperparameters = OrderedDict()
hyperparameters['activation_fn'] = ['sigmoid', 'tanh', 'relu']
hyperparameters['dropout_rate'] = [0, 0.1, 0.3, 0.5]
hyperparameters['reg_param'] = [0, 0.1, 0.01, 0.001]

selected_hyperparam = {hyp:values[0] for hyp, values in hyperparameters.items()}
best_model = None

# In[ ]:


print('Hyperparam Sequential Search')
for hyperparam, values in hyperparameters.items():
    best_value_valacc_seen_so_far = [None, -1]
    for value in values:
        print(f'Hyperparam: {hyperparam}, Value: {value}')
        es = EarlyStopping(monitor='val_acc', mode = 'max', patience = 2, verbose=1)
        passed_hyperparam = copy(selected_hyperparam)
        passed_hyperparam[hyperparam] = value
        model = create_nn_model(**passed_hyperparam)
        model.fit(X_train, y_train,
                  batch_size = _BATCH_SIZE,
                  epochs = _N_EPOCHS,
                  validation_data = (X_val, y_val),
                  callbacks=[es],
                  verbose = 0)
        _, val_acc = model.evaluate(X_val, y_val)
        if val_acc > best_value_valacc_seen_so_far[1]:
            best_value_valacc_seen_so_far[0] = value
            best_value_valacc_seen_so_far[1] = val_acc
            best_model = model
        del model
    selected_hyperparam[hyperparam] = best_value_valacc_seen_so_far[0]

print(f'Selected Hyperparameters: {selected_hyperparam}')


# In[ ]:

'''
print('Retraining Best Model With Selected Hyperparam')
model = create_nn_model(**selected_hyperparam)
model.fit(X_train, y_train,
          batch_size = _BATCH_SIZE,
          epochs = _N_EPOCHS,
          validation_data = (X_val, y_val),
          callbacks=[es])

'''
# In[ ]:


print('Testing Best Model')
loss_acc = best_model.evaluate(X_test, y_test)
print(f'Test Accuracy: {loss_acc[1]}')

