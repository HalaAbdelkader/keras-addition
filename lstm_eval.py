"""
Python script to load a trained LSTM encoder-decoder sequence to sequence model for evaluation.
The input is a variable length sequence of intgeres. The output is the sum of these numbers.
Email: habdelkader@deakin.edu.au

# Usage:
python lstm_eval.py

This script requires Python3, and Keras and Tensorflow deep learning frameworks.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
import numpy as np
import os
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from data_generation import convert_num_to_str, encode_chars_to_integers, one_hot_encode, invert_encoding
from hellinger_distance import get_training_samples, hellinger_distance


# fix the data generation seed for reproducibility
random.seed(1)
np.random.seed(1)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def generate_encode_dataset(n_samples, lower_bound, upper_bound, vocab, in_max_len, out_max_len, max_features=5,
                            data_type='integers'):
    """
       the main method for the variable length random dataset generation and encoding
       for the sequence to sequence LSTM model training and evaluation.
       :param n_samples: number of samples to generate
       :param lower_bound: lower limit for the numbers to be generated
       :param upper_bound: upper limit for the numbers to be generated
       :param vocab: the list of unique characters for this task [0->9, +, '']
       :param in_max_len: the maximum length of output character representation of input sequences
       :param out_max_len: the maximum length of output character representation of target sequences
       :param max_features: the maximum numbers to be added
       :param data_type: the datatype for the dataset to generate 'integers, negatives or float'
       :return: two arrays of variable length sequences for training and evaluating the LSTM model in a
       one-hot encoded representation.
       """

    input = []
    target = []
    if data_type == 'neg':
        if not os.path.exists('lstm_negative_dataset.txt') and not os.path.exists('lstm_negative_dataset_labels.txt'):
            for i in range(n_samples):
                input_seq = [random.randint(lower_bound, upper_bound) for _ in range(max_features)]
                output_seq = np.sum(input_seq)
                input.append(input_seq)
                target.append(output_seq)
            # convert inputs and targets to strings of characters
            input, target = convert_num_to_str(input, target)
            np.savetxt('lstm_negative_dataset.txt', input, fmt="%s")
            np.savetxt('lstm_negative_dataset_labels.txt', target, fmt="%s")
        else:
            input = np.loadtxt('lstm_negative_dataset.txt', dtype='str')
            target = np.loadtxt('lstm_negative_dataset_labels.txt', dtype='str')

    elif data_type == 'floating point':
        if not os.path.exists('lstm_float_dataset.txt') and not os.path.exists('lstm_float_dataset_labels.txt'):
            for i in range(n_samples):
                input_seq = np.round([np.random.uniform(lower_bound, upper_bound) for _ in range(max_features)], 4)
                output_seq = np.sum(input_seq)
                input.append(input_seq)
                target.append(output_seq)
            # convert inputs and targets to strings of characters
            input, target = convert_num_to_str(input, target)
            np.savetxt('lstm_float_dataset.txt', input, fmt="%s")
            np.savetxt('lstm_float_dataset_labels.txt', target, fmt="%s")
        else:
            input = np.loadtxt('lstm_float_dataset.txt', dtype='str')
            target = np.loadtxt('lstm_float_dataset_labels.txt', dtype='str')
    else:
        if not os.path.exists('lstm_positive_dataset.txt') and not os.path.exists('lstm_positive_dataset_labels.txt'):
            for i in range(n_samples):
                input_seq = [random.randint(lower_bound, upper_bound) for _ in range(max_features)]
                output_seq = np.sum(input_seq)
                input.append(input_seq)
                target.append(output_seq)
            # convert inputs and targets to strings of characters
            input, target = convert_num_to_str(input, target)
            np.savetxt('lstm_positive_dataset.txt', input, fmt="%s")
            np.savetxt('lstm_positive_dataset_labels.txt', target, fmt="%s")
        else:
            input = np.loadtxt('lstm_positive_dataset.txt', dtype='str')
            target = np.loadtxt('lstm_positive_dataset_labels.txt', dtype='str')

    # encode characters into indexed integer representation using the vocabulary
    input, target = encode_chars_to_integers(input, target, vocab, in_max_len, out_max_len)

    # vectorisation using one-hot encoding of the indexed integers
    input, target = one_hot_encode(input, target, len(vocab))
    input = np.array(input)
    target = np.array(target)

    return input, target


def main():
    # dataset parameters
    lower_bound = 0
    upper_bound = 100  # range of numbers
    vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ', '.', '-']
    n_chars = len(vocab)
    in_seq_length = 40
    out_seq_length = 4
    n_samples_test = 100
    n_features = 5

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
    #  data_type='neg'
    xtest, ytest = generate_encode_dataset(n_samples_test, lower_bound, upper_bound, vocab, in_seq_length,
                                           out_seq_length) #data_type='floating point')

    # evaluate on the test dataset
    result = model.predict(xtest, verbose=0)

    expected = [invert_encoding(y, vocab) for y in ytest]
    predicted = [invert_encoding(ypred, vocab) for ypred in result]

    expected = np.array(expected).astype(float)
    predicted = np.array(predicted).astype(float)

    rmse = np.sqrt(mean_squared_error(expected, predicted))
    mae = mean_absolute_error(expected, predicted)
    std = np.std(np.abs(expected - predicted))
    print(type(expected))
    print('=> LSTM model results')
    print('=> LSTM: RMSE: {:.6f}, MAE: {:.6f}, STD: {:.6f}'.format(rmse, mae, std))
    print('=> LSTM: Target Prediction Acc: {}'.format(accuracy_score(np.round(expected), predicted)))

    # show some predictions
    print('=> sample model predictions')
    for i in range(10):
        print('Target: {}, Prediction: {}'.format(expected[i], predicted[i]))

    test_dataset = []

    for x in xtest:
        x = invert_encoding(x, vocab)
        x = x.split('+')
        x = np.array(x, dtype=float)
        test_dataset.append(x)

    xdataset, y = get_training_samples(n_features, n_samples_test, upper_bound)
    print("Hellinger distance: ", hellinger_distance(np.array(xdataset), np.array(test_dataset), lower_bound))


# code starting point
if __name__ == '__main__':
    main()


