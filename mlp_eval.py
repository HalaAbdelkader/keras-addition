"""
Python script to load a trained MLP model to perform the linear addition operation for evaluation. 
The input is a length sequence of intgeres. The output is the sum of these numbers.

Email: h.abdelkader@fci-cu.edu.eg

# Usage:
python mlp_inference.py

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
from mlp_train import generate_random_pairs, inverse_normalisation

# fix the data generation seed for reproducibility
random.seed(1)
np.random.seed(1)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def main():
    
    n_samples_test = 10000
    n_features = 10
    largest = 100
    
    # create the MLP model
    print('=> creating MLP model')
    model = Sequential()
    model.add(Dense(20, input_dim=n_features))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    
    print('=> loading mlp model weights')
    model.load_weights('mlp_best_model.h5')

    print('=> sampling test dataset')
    xtest, ytest = generate_random_pairs(n_samples_test, n_features, largest)
    
    # evaluate on the test dataset
    result = model.predict(xtest, verbose=0)
    
    expected = inverse_normalisation(ytest, n_features, largest)
    predicted = inverse_normalisation(result[:, 0], n_features, largest)

    rmse = np.sqrt(mean_squared_error(expected, predicted))
    mae = mean_absolute_error(expected, predicted)
    std = np.std(np.abs(expected - predicted))

    predicted = predicted.astype(int)
    expected = expected.astype(int)

    print('=> MLP: RMSE: {:.6f}, MAE: {:.6f}, STD: {:.6f}'.format(rmse, mae, std))
    print('=> MLP: Target Prediction Acc: {}'.format(accuracy_score(expected, predicted)))
    
    # show some predictions
    print('=> sample model predictions')
    for i in range(20):
        print('Target: {}, Prediction: {}'.format(expected[i], predicted[i]))


# code starting point
if __name__ == '__main__':
    main()


