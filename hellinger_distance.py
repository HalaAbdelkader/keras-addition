"""
Email: habdelkader@deakin.edu.au

# Usage:
python hellinger_distance.py

"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
import numpy as np
import random
import os
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


def get_training_samples(n_features, n_samples=100, upper_bound=100):
    temp = []
    samples = []
    target = []
    if not os.path.exists('test_dataset.txt') and not os.path.exists('test_dataset_labels.txt'):
        for i in range(n_samples):
            input_seq = [random.randint(0, upper_bound) for _ in range(n_features)]
            output_seq = np.sum(input_seq)
            samples.append(input_seq)
            target.append(output_seq)

        np.savetxt('test_dataset.txt', samples, fmt="%s")
        np.savetxt('test_dataset_labels.txt', target, fmt="%s")

    else:
        target = np.loadtxt('test_dataset_labels.txt')
        temp = np.loadtxt('test_dataset.txt', delimiter='\n', dtype=str)

        for i in range(len(temp)):
            sequence = temp[i].split(' ')
            samples.append([float(element) for element in sequence])
        samples = np.array(samples)
        target = np.array(target)

    return samples, target


def hellinger_distance(vec1, vec2, lower_bound, max_features=5):
    """
    a helper function that calculate Hellinger distance between two probability distributions.
    :param vec1: numpy.ndarray distribution vector
    :param vec2: numpy.ndarray distribution vector
    :param lower_bound: lower limit for the numbers generated
    :param max_features: number of samples generated
    :return:  float Hellinger distance between `vec1` and `vec2`
    Value in range [0, 1], where 0 is min distance (max similarity) and 1 is max distance (min similarity).
    """

    vec1 = vec1 - lower_bound
    vec2 = vec2 - lower_bound

    # normalize the datasets
    vec1 = vec1/(100.0 * max_features)  # 100 upper bound
    vec2 = vec2/(100.0 * max_features)
    sim = np.sqrt(0.5 * ((np.sqrt(vec1) - np.sqrt(vec2))**2).sum(axis=1))
    sim = sim.mean()
    return sim
