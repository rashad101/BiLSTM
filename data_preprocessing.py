"""
@author: Md Rashad Al Hasan Rony

"""

import numpy as np
import pandas as pd
import tensorflow as tf

titles = ["id", "name", "description"]


def to_one_hot(y, n_class):
    return np.eye(n_class)[y.astype(int)]


def load_data(file_name, sample_ratio= 1, n_class=15, names=titles):
    csv_file = pd.read_csv(file_name, names=names)
    shuffled = csv_file.sample(frac=sample_ratio)
    x = pd.Series(shuffled["description"])
    y = pd.Series(shuffled["id"])
    y = to_one_hot(y, n_class)
    return x, y

def process_data(train, test, max_len):
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
    x_transform_train = vocab_processor.fit_transform(train)
    x_transform_test = vocab_processor.transform(test)
    vocab = vocab_processor.vocabulary_
    vocab_size = len(vocab)
    x_train_list = list(x_transform_train)
    x_test_list = list(x_transform_test)
    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)
    return x_train, x_test, vocab, vocab_size

def split_data(x_test, y_test, ratio):
    test_size = len(x_test)
    val_size = (int)(test_size * ratio)
    x_val = x_test[:val_size]
    x_test = x_test[val_size:]
    y_val = y_test[:val_size]
    y_test = y_test[val_size:]
    return x_test, x_val, y_test, y_val, val_size, test_size - val_size
