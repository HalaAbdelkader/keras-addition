"""
Python script to train a sequence to sequence learning LSTM model to perform the linear addition operation.
By: Hala Abdelkader
Email: habdelkader@deakin.edu.au

# Usage:
python lstm_train.py

This script requires Python3, and Keras and Tensorflow deep learning frameworks.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
import random
# fix the data generation seed for reproducibility
random.seed(1)
np.random.seed(1)
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from data_generation import generate_save_dataset, encode_dataset

# Configure Tensorflow and Keras backends seed
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


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
    plt.close()


def main(args):
    # dataset parameters
    n_samples_train = 100000
    largest = 100  # upper range of numbers i.e, [0, 100]
    batch_size = 128
    vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ', '.', '-']
    n_chars = len(vocab)

    in_seq_length = 40  # sufficient to represent up to variable length sequences of 10 characters
    out_seq_length = 4  # sufficient to represent the output of adding up to 95 integers of range [0, 100]

    # define LSTM configuration
    num_epochs = 1000
    xdataset = []
    ylabel = []
    max_features = 10

    if not os.path.exists(args.xdata) or not os.path.exists(args.ylabel):
        xdataset, ylabel = generate_save_dataset(n_samples_train, largest, max_features)

    else:
        xdataset = np.loadtxt(args.xdata, dtype='str')
        ylabel = np.loadtxt(args.ylabel, dtype='str')

    xdataset, ylabel = encode_dataset(xdataset, ylabel, vocab, in_seq_length, out_seq_length)
    # 10 fold cross validation
    kf = KFold(n_splits=10)
    k = 0
    best_scores = []
    for train_index, test_index in kf.split(xdataset):
        xtrain, xtest, ytrain, ytest = xdataset[train_index], xdataset[test_index], ylabel[train_index], ylabel[test_index]

        # create LSTM Model
        model = Sequential()
        model.add(LSTM(128, input_shape=(None, n_chars)))
        model.add(RepeatVector(out_seq_length))
        model.add(LSTM(128, return_sequences=True))
        model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        checkpoint = ModelCheckpoint('lstm_update_best_model_kf_{}.h5'.format(k), verbose=1, monitor='val_loss',
                                     save_best_only=True, mode='min')

        # train LSTM
        H = None
        train_losses = []
        val_losses = []
        train_acc = []
        val_acc = []

        f = open('lstm_results_k_{}.txt'.format(k), 'w')
        for i in range(num_epochs):
            print('Epoch: {}/{}'.format(i, num_epochs))
            H = model.fit(xtrain, ytrain, epochs=1, batch_size=batch_size, validation_data=(xtest, ytest),
                          callbacks=[checkpoint])

            print(H.history)
            train_losses.append(H.history["loss"][0])
            val_losses.append(H.history["val_loss"][0])
            train_acc.append(H.history["accuracy"][0])
            val_acc.append(H.history["val_accuracy"][0])

            #plot curves every 10 epochs
            if i % 10 == 0 and i > 0:
                plot_training_curves(train_losses, val_losses)

            f.write('{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(i, train_losses[-1], train_acc[-1], val_losses[-1], val_acc[-1]))
            f.write('\n')
            f.flush()
        f.close()
        k = k + 1
        best_scores.append(np.max(val_acc))


    print("==>> Cross validation finished")
    for k, s in enumerate(best_scores):
        print('==> Fold {} validation accuracy: {}'.format(k, s))
    print('==> Cross validation average accuracy: {}'.format(np.mean(best_scores)))


# code starting point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mathematical Addition LSTM Training')
    parser.add_argument('-x', '--xdata', metavar='DIR', default="xdataset.txt", help='path to dataset inputs')
    parser.add_argument('-y', '--ylabel', metavar='DIR', default="ylabel.txt", help='path to dataset targets')
    args = parser.parse_args()
    main(args)
