"""
Python script to load a trained MLP model to perform the linear addition operation for evaluation. 
The input is a length sequence of intgeres. The output is the sum of these numbers.

Email: habdelkader@deakin.edu.au

# Usage:
python mlp_eval.py

This script requires Python3, and Keras and Tensorflow deep learning frameworks.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score


# fix the data generation seed for reproducibility
random.seed(1)
np.random.seed(1)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def inverse_normalisation(x, n_features, largest):
    return x * float(largest * n_features)


def generate_integer_random_pairs(n_samples, n_features, lower_bound, upper_bound, data_type='integers'):
    """
    a helper function that generates a dataset of random numbers with their sum as labels
    :param n_samples: number of samples to generate
    :param n_features: number of elements per sample
    :param lower_bound: the lower bound of the random sampling range
    :param upper_bound: the upper bound of the random sampling range
    :param data_type: positive integers, negative integers, positive floating points, negative floating points
    :return: normalised dataset of numbers and labels
    """

    if data_type == 'float_numbers':
        input = np.random.uniform(lower_bound, upper_bound, [n_samples, n_features])
        target = np.sum(input, axis=1)
    else:
        input = np.random.randint(lower_bound, upper_bound, [n_samples, n_features])
        target = np.sum(input, axis=1)

    # normalize
    x = input.astype('float') / float(upper_bound * n_features)
    y = target.astype('float') / float(upper_bound * n_features)
    return x, y


def model_creation(n_features):
    # create the MLP model
    print('=> creating MLP model')
    model = Sequential()
    model.add(Dense(20, input_dim=n_features))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model


def evaluate_model(expected, predicted):
    rmse = np.sqrt(mean_squared_error(expected, predicted))
    mae = mean_absolute_error(expected, predicted)
    std = np.std(np.abs(expected - predicted))

    print('=> MLP: RMSE: {:.6f}, MAE: {:.6f}, STD: {:.6f}'.format(rmse, mae, std))
    print('=> MLP: Target Prediction Acc: {}'.format(accuracy_score(np.round(expected), np.round(predicted))))

    # show some predictions
    print('=> sample model predictions')
    for i in range(20):
        print('Target: {}, Prediction: {}'.format(expected[i], predicted[i]))


def main():
    
    n_samples_test = 1000
    n_features = 10
    lower_bound = -100
    upper_bound = 100

    model = model_creation(n_features)

    print('=> loading mlp model weights')
    model.load_weights('mlp_best_model.h5')

    print('=> sampling test dataset')

    # Positive (lower_bound = 0) and negative integers (lower_bound = -100)
    # xtest, ytest = generate_integer_random_pairs(n_samples_test, n_features, lower_bound, upper_bound)

    # Positive (lower_bound = 0) and negative (lower_bound = -100) floating points
    xtest, ytest = generate_integer_random_pairs(n_samples_test, n_features, lower_bound, upper_bound, 'float_numbers')
    
    # evaluate on the test dataset
    result = model.predict(xtest, verbose=0)
    expected = inverse_normalisation(ytest, n_features, upper_bound)
    predicted = inverse_normalisation(result[:, 0], n_features, upper_bound)
    evaluate_model(expected, predicted)


# code starting point
if __name__ == '__main__':
    main()


