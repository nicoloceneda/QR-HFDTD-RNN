""" htqf_rnn.py
    -----------
    This script constructs the heavy tail quantile function via a recurrent neural network.

    Contact: nicolo.ceneda@student.unisg.ch
    Last update: 25 June 2019

"""


# ------------------------------------------------------------------------------------------------------------------------------------------
# PROGRAM SETUP
# ------------------------------------------------------------------------------------------------------------------------------------------


# Import the libraries and the modules

import pandas as pd
import numpy as np
import scipy.stats

import os
import sys

import tensorflow as tf
from datetime import datetime


symbol_list = ['AAPL'] # TODO: fix data input


epochs = 5 # TODO: make epochs sttable

def allocate_batch(length_):

    batch_list = [[] for i in range(epochs)]

    pos = 0

    for i in range(length_):

        batch_list[pos].append(i)
        pos = pos+1 if pos+1 < epochs else 0

    return batch_list


l = 100 # TODO: make l sttable

for symbol in symbol_list:

    folder = '{}/{}/'.format(symbol, l)
    n_features = 4

    X_train = pd.read_csv(folder + 'X_train.csv').values.reshape(-1, l, n_features)
    X_valid = pd.read_csv(folder + 'X_valid.csv').values.reshape(-1, l, n_features)
    X_test = pd.read_csv(folder + 'X_test.csv').values.reshape(-1, l, n_features)

    Y_train = pd.read_csv(folder + 'Y_train.csv').values
    Y_valid = pd.read_csv(folder + 'Y_valid.csv').values
    Y_test = pd.read_csv(folder + 'Y_test.csv').values

    batch_list = allocate_batch(Y_train.shape[0])

    tau_full = np.concatenate(([0.01], np.arange(0.05, 1, 0.05), [0.99])).reshape(1, -1)
    tau_full_perc = scipy.stats.norm.ppf(tau_full)

    tau_new = np.array([0.01, 0.05, 0.10]).reshape(1, -1)
    tau_new_perc = scipy.stats.normal.ppf(tau_new)

    A = 4
    hidden_layer_dim = 16
    output_layer_dim = 4






























