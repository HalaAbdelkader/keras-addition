"""
Python script to train an MLP model to perform the linear addition operation.
By: Hala Abdelkader
Email: h.abdelkader@fci-cu.edu.eg

# Usage:
python mlp_train.py

"""

from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
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


def generate_random_pairs(n_samples, n_features, range):
    """
    a helper function that generates a dataset of random numbers with their sum as labels
    :param n_samples: number of samples to generate
    :param n_features: number of elements per sample
    :param range: range of numbers to sample from
    :return: normalised dataset of numbers and labels
    """
    input = np.random.randint(0, range, [n_samples, n_features])
    target = np.sum(input, axis=1)

    # normalize
    x = input.astype('float') / float(range * n_features)
    y = target.astype('float') / float(range * n_features)
    return x, y


def inverse_normalisation(x, n_features, largest):
    return np.round(x * float(largest * n_features))


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


def main():
    # dataset parameters

    n_samples_train = 1000
    n_samples_test = 200
    n_features = 10  # number of elements to add
    largest = 100   # range of numbers

    Xtrain, ytrain = generate_random_pairs(n_samples_train, n_features, largest)
    Xtest, ytest = generate_random_pairs(n_samples_test, n_features, largest)

    num_epochs = 100  # number of epochs to train

    # create the MLP model
    model = Sequential()
    model.add(Dense(20, input_dim=n_features))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    print(model.summary())

    # train the model
    H = model.fit(Xtrain, ytrain, epochs=num_epochs, batch_size=32, verbose=1, validation_split=0.2,
                  validation_data=(Xtest, ytest))
    model.save_weights('mlp_best_model.h5')    
    plot_training_curves(H)

    # evaluate on the test dataset
    result = model.predict(Xtest, verbose=0)
    
    expected = inverse_normalisation(ytest, n_features, largest)
    predicted = inverse_normalisation(result[:, 0], n_features, largest)

    rmse = np.sqrt(mean_squared_error(expected, predicted))
    mae = mean_absolute_error(expected, predicted)
    std = np.std(np.abs(expected - predicted))

    predicted = predicted.astype(int)
    expected = expected.astype(int)
    print('=> MLP model results')
    print('=> MLP: RMSE: {:.6f}, MAE: {:.6f}, STD: {:.6f}'.format(rmse, mae, std))
    print('=> MLP: Target Prediction Acc: {}'.format(accuracy_score(expected, predicted)))
    
    # show some predictions
    print('=> sample model predictions')
    for i in range(20):
        print('Target: {}, Prediction: {}'.format(expected[i], predicted[i]))


if __name__ == '__main__':
    main()
