""" lstm_rnn.py
    -----------
    This script executes a long short term memory recurrent neural network to estimate the parameters of a parametric heavy tailed quantile
    function.

    Contact: nicolo.ceneda@student.unisg.ch
    Last update: 18 May 2020
"""


# ------------------------------------------------------------------------------------------------------------------------------------------
# 0. CODE SETUP
# ------------------------------------------------------------------------------------------------------------------------------------------


# Import the libraries

import sys
import os
import numpy as np
import pandas as pd
import scipy.stats
import tensorflow as tf
from datetime import datetime


# Set the seed

tf.random.set_seed(1)


# ------------------------------------------------------------------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# ------------------------------------------------------------------------------------------------------------------------------------------


# Import the extracted datasets

symbol = 'AAPL'
data_extracted = pd.read_csv('mode sl/datasets/' + symbol + '/data.csv')


# Create the features and target datasets

log_return = np.diff(np.log(data_extracted['price']))
data = pd.DataFrame({'log_return': log_return})

elle = 100
data['log_return_ma'] = data['log_return'].rolling(window=elle).mean()

X = []
Y = []

for pos in range(elle, data.shape[0]):

    Y.append(data.iloc[pos]['log_return'])

    data_past = pd.DataFrame(data.iloc[pos - elle: pos], copy=True)
    r_past = data_past['log_return']
    r_past_ma = data_past.iloc[-1]['log_return_ma']
    r_diff = r_past - r_past_ma
    data_past['log_return_d2'] = r_diff ** 2
    data_past['log_return_d3'] = r_diff ** 3
    data_past['log_return_d4'] = r_diff ** 4
    X.append(data_past[['log_return', 'log_return_d2', 'log_return_d3', 'log_return_d4']])

X = pd.concat(X, ignore_index=True)
Y = pd.DataFrame(Y, columns=['label'])


# Define the training, validation and test subsets

train_split = int(Y.shape[0] * 0.8)
valid_split = train_split + int(Y.shape[0] * 0.1)
test_split = train_split + int(Y.shape[0] * 0.1) + int(Y.shape[0] * 0.1)

X_train = pd.DataFrame(X.iloc[:train_split * elle], copy=True)
Y_train = pd.DataFrame(Y.iloc[:train_split], copy=True)

X_valid = pd.DataFrame(X.iloc[train_split * elle: valid_split * elle], copy=True)
Y_valid = pd.DataFrame(Y.iloc[train_split: valid_split], copy=True)

X_test = pd.DataFrame(X.iloc[valid_split * elle: test_split * elle], copy=True)
Y_test = pd.DataFrame(Y.iloc[valid_split: test_split], copy=True)


# Standardize the training, validation and test subsets

for column in X_train.columns:

    column_mean = X_train[column].mean()
    column_std = X_train[column].std()

    X_train[column] = (X_train[column] - column_mean) / column_std
    X_valid[column] = (X_valid[column] - column_mean) / column_std
    X_test[column] = (X_test[column] - column_mean) / column_std

Y_mean = Y_train.mean()
Y_std = Y_train.std()

Y_train = (Y_train - Y_mean) / Y_std
Y_valid = (Y_valid - Y_mean) / Y_std
Y_test = (Y_test - Y_mean) / Y_std


# Reshape the train, validation and test subsets

n_features = 4

X_train = X_train.values.reshape(-1, elle, n_features)
X_valid = X_valid.values.reshape(-1, elle, n_features)
X_test = X_test.values.reshape(-1, elle, n_features)

Y_train = Y_train.values
Y_valid = Y_valid.values
Y_test = Y_test.values


# Create train, validation and test tensorflow datasets

batch_size = 300

ds_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
ds_train = ds_train.cache().shuffle(10000).batch(batch_size).repeat()

ds_valid = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
ds_valid = ds_valid.batch(batch_size).repeat()

ds_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
ds_test = ds_test.batch(batch_size).repeat()

for batch in ds_train.take(1):

    array_features = batch[0]
    array_targets = batch[1]

    print('The dataset is made up of several batches, each containing:'
          '\n- An array of 300 time series of features'
          '\n- An array of 300 targets'
          '\n'
          '\nBatch 0'
          '\n-------'
          '\nTuple 0\n',
          array_features.numpy()[0], array_targets.numpy()[0],
          '\nTuple 300\n',
          array_features.numpy()[299], array_targets.numpy()[299])


# ------------------------------------------------------------------------------------------------------------------------------------------
# 2. DESIGN THE MODEL
# ------------------------------------------------------------------------------------------------------------------------------------------


tau = np.concatenate(([0.01], np.divide(range(1, 20), 20), [0.99])).reshape(1, -1)
new_tau = np.array([0.01, 0.05, 0.1]).reshape(1, -1)
quantile_standard_normal = scipy.stats.norm.ppf(tau, loc=0.0, scale=1.0)
new_quantile_standard_normal = scipy.stats.norm.ppf(new_tau, loc=0.0, scale=1.0)

A = 4
n_hidden = 16 # TODO: = int(sys.argv[3])
n_outputs = 4







