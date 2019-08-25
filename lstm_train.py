"""
Python script to train a sequence to sequence learning LSTM model to perform the linear addition operation.
By: Hala Abdelkader
Email: h.abdelkader@fci-cu.edu.eg

# Usage:
python lstm_train.py

This script requires Python3, and Keras and Tensorflow deep learning frameworks.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
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


# fix the data generation seed for reproducibility
random.seed(1)
np.random.seed(1)

# Configure Tensorflow and Keras backends seed
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def generate_encode_dataset(n_samples, largest, vocab, in_max_len, out_max_len, max_features=10):
    """
    the main method for the variable length random dataset generation and encoding
    for the sequence to sequence LSTM model training and evaluation.
    :param n_samples: number of samples to generate
    :param largest: upper range for the numbers to be generated, lower is zero
    :param vocab: the list of unique characters for this task [0->9, +, '']
    :param in_max_len: the maximum length of output character representation of input sequences
    :param out_max_len: the maximum length of output character representation of target sequences
    :return: two arrays of variable length sequences for training and evaluating the LSTM model in a
    one-hot encoded representation.
    """
    # generate variable length numeric input-target pairs
    input, target = generate_random_pairs(n_samples, largest, max_features)

    # convert inputs and targets to strings of characters
    input, target = convert_num_to_str(input, target, in_max_len, out_max_len)

    # encode characters into indexed integer representation using the vocabulary
    input, target = encode_chars_to_integers(input, target, vocab)

    # vectorisation using one-hot encoding of the indexed integers
    input, target = one_hot_encode(input, target, len(vocab))

    input = np.array(input)
    target = np.array(target)

    return input, target


def generate_random_pairs(n_samples, largest, max_features):
    """
    a helper function that generates a dataset of variable length sequences of random numbers with their sum as labels
    :param n_samples: number of samples to generate
    :param largest: upper range for the numbers to be generated, lower is zero
    :param max_features: optional parameter in case a fixed sequence is needed
    :return: two numpy arrays for the dataset sequences and labels
    """

    input = []
    target = []

    for i in range(n_samples):
        n_features = random.randint(2, max_features)
        input_seq = [random.randint(0, largest) for _ in range(n_features)]
        output_seq = np.sum(input_seq)
        input.append(input_seq)
        target.append(output_seq)

    input = np.array(input)
    target = np.array(target)

    return input, target


def convert_num_to_str(input, target, in_max_len, out_max_len):
    """
    converts a dataset of sequences of numbers to strings of characters
    :param input: the input sequences i.e, [10+23]
    :param target: the label of the input sequence, i,e. [33]
    :param in_max_len: the maximum length of output character representation of input sequence
    :param out_max_len: the maximum length of output character representation of targets
    :return: the intput and output sequences as string of characters with leading spaces padded if necessary
    """
    input_str = []
    output_str = []

    for sequence in input:
        xstr = '+'.join([str(elem) for elem in sequence])
        xstr = ''.join([' ' for _ in range(in_max_len - len(xstr))]) + xstr
        input_str.append(xstr)

    for sequence in target:
        ystr = str(sequence)
        ystr = ''.join([' ' for _ in range(out_max_len - len(ystr))]) + ystr
        output_str.append(ystr)
    return input_str, output_str


def encode_chars_to_integers(input, target, vocab):
    """
    encodes the characters of the string sequence into integer representation based the index of the character
    in the vocab. i.e., ['33+24'] -> [4, 4, 11, 3, 5]
    :param input: array of input string sequences
    :param target: array of target output string sequences
    :param vocab: the list of unique characters for this task [0->9, +, '']
    :return: two arrays of indexed integer representations of input and target sequences
    """
    encoded_input = []
    encoded_target = []

    for xstr in input:
        int_encoded_seq = [vocab.index(c) for c in xstr]
        encoded_input.append(int_encoded_seq)

    for ystr in target:
        int_encoded_seq = [vocab.index(c) for c in ystr]
        encoded_target.append(int_encoded_seq)
    return encoded_input, encoded_target


def one_hot_encode(input, target, vocab_length):
    """
    vectorisation function to convert the indexed integer representation [4, 4, 11, 3, 5] into vectors suitable
    for LSTM training. i.e, 4 will be converted to [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    :param input:  array of indexed integer input sequences
    :param target: array of indexed integer target sequences
    :param vocab_length: the length of the vocabulary.
    :return: two arrays of one-hot encoded representations for the input and target sequences
    """
    input_hot_encoded = []
    target_hot_encoded = []

    for sequence in input:
        list_one_hots = []
        for idx in sequence:
            encoded_elem = [0] * vocab_length
            encoded_elem[idx] = 1
            list_one_hots.append(encoded_elem)

        input_hot_encoded.append(list_one_hots)

    for sequence in target:
        list_one_hots = []
        for idx in sequence:
            encoded_elem = [0] * vocab_length
            encoded_elem[idx] = 1
            list_one_hots.append(encoded_elem)

        target_hot_encoded.append(list_one_hots)

    return input_hot_encoded, target_hot_encoded


def invert_encoding(seq, vocab):
    """
    a helper function to inverse the predicted sequence of character probabilities to a string of characters
    :param seq: sequence of character probabilities
    :param vocab: the list of unique characters for this task [0->9, +, '']
    :return: string representation of probabilities, i.e, [[0.01, 0.7, 0.2,,]..] -> ['1..']
    """
    int_to_char = dict((i, c) for i, c in enumerate(vocab))
    strings = []
    for pattern in seq:
        string = int_to_char[np.argmax(pattern)]
        strings.append(string)
    return ''.join(strings).strip()


def plot_training_curves(train_loss, val_loss):
    """
    a helper function to plot and saves the training details
    :param train_loss: a list of training losses
    :param val_loss: a list of validation losses
    :return: nothing
    """
    N = len(train_loss)
    plt.figure()
    plt.plot(np.arange(0, N), train_loss, label="train_loss")
    plt.plot(np.arange(0, N), val_loss, label="val_loss")

    plt.grid()
    plt.title("Training and Validation Loss ")
    plt.xlabel("Epoch")
    plt.ylabel("Loss MSE")
    plt.legend(loc="upper right")
    plt.savefig("lstm_plot.png")


def main():

    # dataset parameters
    n_samples_train = 5000
    n_samples_test = 100
    largest = 100  # upper range of numbers i.e, [0, 100]
    batch_size = 128
    vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
    n_chars = len(vocab)

    in_seq_length = 40  # sufficient to represent up to variable length sequences of 10 characters
    out_seq_length = 4  # sufficient to represent the output of adding up to 95 integers of range [0, 100]

    # define LSTM configuration
    num_epochs = 11

    # create LSTM
    model = Sequential()
    model.add(LSTM(128, input_shape=(None, n_chars)))
    model.add(RepeatVector(out_seq_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # train LSTM
    H = None
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    best_loss = 100000
    f = open('lstm_results.txt', 'w') 
    for i in range(num_epochs):
        print('Epoch: {}/{}'.format(i, num_epochs))
        # instead of memorising the training dataset, sample a new dataset at each epoch and train on it.
        xtrain, ytrain = generate_encode_dataset(n_samples_train, largest, vocab, in_seq_length, out_seq_length)
        # use 20% of the randomly sampled dataset for validation.
        H = model.fit(xtrain, ytrain, epochs=1, batch_size=batch_size, validation_split=0.2)

        train_losses.append(H.history["loss"][0])
        val_losses.append(H.history["val_loss"][0])
        train_acc.append(H.history["acc"][0])
        val_acc.append(H.history["val_acc"][0])

        # save the best model
        cur_val_loss = H.history["val_loss"][0]
        if cur_val_loss < best_loss:
            print('* best model: ', cur_val_loss)
            model.save_weights('lstm_best_model.h5')
            best_loss = cur_val_loss
        # plot curves every 10 epochs
        if i % 10 == 0 and i > 0:
            plot_training_curves(train_losses, val_losses)
    
        # log the training results        
        f.write('{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(i, train_losses[-1], train_acc[-1], val_losses[-1], val_acc[-1]))
        f.write('\n')
        f.flush()
    f.close()

    plot_training_curves(train_losses, val_losses)

    print('==> finished training for {} epochs.'.format(num_epochs))
    print('==> loading the best checkpoint to test.')
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
