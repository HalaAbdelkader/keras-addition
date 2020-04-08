"""
Python script to train an MLP model to perform the linear addition operation.
By: Hala Abdelkader
Email: habdelkader@deakin.edu.au

# Usage:
python mlp_train.py

"""

from __future__ import print_function
import numpy as np
import os
import random
import argparse
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from data_generation import generate_save_dataset, normalisation, inverse_normalisation


# fix the data generation seed for reproducibility
random.seed(1)
np.random.seed(1)
# Configure Tensorflow and Keras backends seed
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def plot_training_curves(H):
    """
    plots and saves the training details
    :param H: model history object
    :return: nothing
    """
    N = len(H.history["loss"])
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.grid()
    plt.title("Training and Validation Loss ")
    plt.xlabel("Epoch")
    plt.ylabel("Loss MSE")
    plt.legend(loc="upper right")
    plt.savefig("mlp_plot.png")


def main(args):

    # dataset parameters
    n_samples_train = 100000
    n_features = 10  # number of elements to add
    largest = 100  # range of numbers
    num_epochs = 100  # number of epochs to train
    rmse_models = []
    mae_models = []
    std_models = []
    model_index = 0
    training_dataset = np.zeros((100000, 10))

    if not os.path.exists(args.xdata) or not os.path.exists(args.ylabel):
        Xdata, ylabel = generate_save_dataset(n_samples_train, largest, n_features)

    else:
        Xdata = np.loadtxt('xdataset.txt',  dtype='str')
        ylabel = np.loadtxt('ylabel.txt')

    for indx, x in enumerate(Xdata):
        x = x.split('+')
        x = np.array(x, dtype=float)
        for i in range(len(x)):
            training_dataset[indx, i] = x[i]

    training_dataset = np.array(training_dataset)
    ylabel = np.array(ylabel)
    training_dataset, ylabel = normalisation(training_dataset, ylabel, largest, n_features)

    # 10-fold cross validation
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(training_dataset):
        X_train, X_test = training_dataset[train_index], training_dataset[test_index]
        y_train, y_test = ylabel[train_index], ylabel[test_index]

        # create the MLP model
        model = Sequential()
        model.add(Dense(20, input_dim=n_features))
        model.add(Dense(10))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())

        # train the model
        H = model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, verbose=1,
                      validation_data=(X_test, y_test))
        model.save_weights('mlp_model'+str(model_index)+'.h5')
        model_index += 1
        plot_training_curves(H)

        # evaluate on the test dataset
        result = model.predict(X_test, verbose=0)

        expected = inverse_normalisation(y_test, n_features, largest)
        predicted = inverse_normalisation(result[:, 0], n_features, largest)

        rmse = np.sqrt(mean_squared_error(expected, predicted))
        mae = mean_absolute_error(expected, predicted)
        std = np.std(np.abs(expected - predicted))

        rmse_models.append(rmse)
        mae_models.append(mae)
        std_models.append(std)

        predicted = predicted.astype(int)
        expected = expected.astype(int)
        print('=> MLP model results')
        print('=> MLP: RMSE: {:.6f}, MAE: {:.6f}, STD: {:.6f}'.format(rmse, mae, std))
        print('=> MLP: Target Prediction Acc: {}'.format(accuracy_score(expected, predicted)))

        # show some predictions
        print('=> sample model predictions')
        for i in range(20):
            print('Target: {}, Prediction: {}'.format(expected[i], predicted[i]))

    best_model = rmse_models.index(min(rmse_models))
    print(best_model)
    model.load_weights('mlp_model'+str(best_model)+'.h5')
    model.save_weights('mlp_best_model.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mathematical Addition MLP Training')
    parser.add_argument('-x', '--xdata', metavar='DIR', default="xdataset.txt", help='path to dataset inputs')
    parser.add_argument('-y', '--ylabel', metavar='DIR', default="ylabel.txt", help='path to dataset targets')
    args = parser.parse_args()
    main(args)