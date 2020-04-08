"""
Python script to generate datasets for the MLP and lstm models to perform the linear addition operation.
By: Hala Abdelkader
Email: habdelkader@deakin.edu.au

# Usage:
python data_generation.py

"""

from __future__ import print_function
import numpy as np
import random
import tensorflow as tf
from keras import backend as K

# fix the data generation seed for reproducibility
random.seed(1)
np.random.seed(1)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


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


def convert_num_to_str(input, target):
    """
    converts a dataset of sequences of numbers to strings of characters
    :param input: the input sequences i.e, [10 23]
    :param target: the label of the input sequence, i,e. [33]
    :return: the intput and output sequences as string of characters with sign + added to the input sequence
    """
    input_str = []
    output_str = []

    for sequence in input:
        xstr = '+'.join([str(elem) for elem in sequence])
        input_str.append(xstr)

    for sequence in target:
        ystr = str(sequence)
        output_str.append(ystr)
    return input_str, output_str


def generate_save_dataset(n_samples, largest, max_features):
    Xdata, ylabel = generate_random_pairs(n_samples, largest, max_features)
    Xdata, ylabel = convert_num_to_str(Xdata, ylabel)
    np.savetxt('lstm_xdataset.txt', Xdata, fmt="%s")
    np.savetxt('lstm_labels.txt', ylabel, fmt="%s")
    return Xdata, ylabel


def normalisation(input, target, upper_bound, n_features):
    # normalize
    x = input.astype('float') / float(upper_bound * n_features)
    y = target.astype('float') / float(upper_bound * n_features)
    return x, y


def inverse_normalisation(x, n_features, largest):
    return np.round(x * float(largest * n_features))


def encode_dataset(input, target, vocab, in_max_len, out_max_len):
    """
    the main method for the variable length random dataset generation and encoding
    for the sequence to sequence LSTM model training and evaluation.
    :param n_samples: number of samples to generate
    :param largest: upper range for the numbers to be generated, lower is zero
    :param vocab: the list of unique characters for this task [0->9, +, '']
    :param in_max_len: the maximum length of output character representation of input sequences
    :param out_max_len: the maximum length of output character representation of target sequences
    :param max_features: the maximum numbers to be added
    :param load: a flag used to determine if we need to generate the dataset or load it if exits
    :return: two arrays of variable length sequences for training and evaluating the LSTM model in a
    one-hot encoded representation.
    """

    # encode characters into indexed integer representation using the vocabulary
    input, target = encode_chars_to_integers(input, target, vocab, in_max_len, out_max_len)

    # vectorisation using one-hot encoding of the indexed integers
    input, target = one_hot_encode(input, target, len(vocab))
    input = np.array(input)
    target = np.array(target)
    return input, target


def encode_chars_to_integers(input, target, vocab, in_max_len, out_max_len):
    """
    encodes the characters of the string sequence into integer representation based the index of the character
    in the vocab. i.e., ['33+24'] -> [4, 4, 11, 3, 5]
    :param input: array of input string sequences
    :param target: array of target output string sequences
    :param vocab: the list of unique characters for this task [0->9, '+', ' ', '.', '-']
    :param in_max_len: the maximum length of output character representation of input sequence
    :param out_max_len: the maximum length of output character representation of targets
    :return: two arrays of indexed integer representations of input and target sequences
    """
    encoded_input = []
    encoded_target = []

    # Leading spaces padded if necessary
    for xstr in input:
        xstr = ''.join([' ' for _ in range(in_max_len - len(xstr))]) + xstr
        int_encoded_seq = [vocab.index(c) for c in xstr]
        encoded_input.append(int_encoded_seq)

    for ystr in target:
        ystr = ''.join([' ' for _ in range(out_max_len - len(ystr))]) + ystr
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