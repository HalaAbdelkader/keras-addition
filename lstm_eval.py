"""
Python script to load a trained LSTM encoder-decoder sequence to sequence model for evaluation.
The input is a variable length sequence of intgeres. The output is the sum of these numbers.
Email: h.abdelkader@fci-cu.edu.eg

# Usage:
python lstm_inference.py

This script requires Python3, and Keras and Tensorflow deep learning frameworks.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
import math
import os
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from keras.optimizers import Adam
from keras.models import load_model
from lstm_train import generate_encode_dataset, invert_encoding


# fix the data generation seed for reproducibility
random.seed(1)
np.random.seed(1)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def main():
    # dataset parameters
    largest = 100  # range of numbers
    vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
    n_chars = len(vocab)    
    in_seq_length = 40 
    out_seq_length = 4 
    n_samples_test = 10000
    # create LSTM
    print('==> creating LSTM model')
    model = Sequential()
    model.add(LSTM(128, input_shape=(None, n_chars)))
    model.add(RepeatVector(out_seq_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    print('==> loading LSTM model weights')
    model.load_weights('lstm_best_model.h5')

    print('==> sampling a test dataset of {} samples.'.format(n_samples_test))
    xtest, ytest = generate_encode_dataset(n_samples_test, largest, vocab, in_seq_length, out_seq_length)

    # evaluate on the test dataset
    result = model.predict(xtest, verbose=0)

    expected = [invert_encoding(y, vocab) for y in ytest]
    predicted = [invert_encoding(ypred, vocab) for ypred in result]

    expected = np.array(expected).astype(int)
    predicted = np.array(predicted).astype(int)
    
    rmse = np.sqrt(mean_squared_error(expected, predicted))
    mae = mean_absolute_error(expected, predicted)
    std = np.std(np.abs(expected - predicted))

    print('=> LSTM model results')
    print('=> LSTM: RMSE: {:.6f}, MAE: {:.6f}, STD: {:.6f}'.format(rmse, mae, std))
    print('=> LSTM: Target Prediction Acc: {}'.format(accuracy_score(expected, predicted)))
    
    # show some predictions
    print('=> sample model predictions')
    for i in range(10):
        print('Target: {}, Prediction: {}'.format(expected[i], predicted[i]))


# code starting point
if __name__ == '__main__':
    main()


