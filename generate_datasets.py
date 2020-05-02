""" generate_datasets.py
    --------------------
    This script generates the train, validation and test sets.

    Parameters to change:
    - symbol
    - elle

    Contact: nicolo.ceneda@student.unisg.ch
    Last update: 18 May 2020
"""


# -------------------------------------------------------------------------------
# 0. CODE SETUP
# -------------------------------------------------------------------------------


# Import the libraries

import os
import sys
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the extracted datasets

symbol = 'NFLX'  # TODO: change back to sys.argv[1]
data_extracted = pd.read_csv('data/mode sl/datasets/' + symbol + '/data.csv')


# Create the features and target datasets

log_return = np.diff(np.log(data_extracted['price']))
data = pd.DataFrame({'log_return': log_return})

elle = 200  # TODO: change back to int(sys.argv[2])
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


# Save the standardized returns

if not os.path.isdir('data/mode sl/datasets std'):

    os.mkdir('data/mode sl/datasets std')

symbol_elle = symbol + '_' + str(elle)

if not os.path.isdir('data/mode sl/datasets std/' + symbol_elle):

    os.mkdir('data/mode sl/datasets std/' + symbol_elle)

X_train.to_csv('data/mode sl/datasets std/' + symbol_elle + '/X_train.csv', index=False)
Y_train.to_csv('data/mode sl/datasets std/' + symbol_elle + '/Y_train.csv', index=False)

X_valid.to_csv('data/mode sl/datasets std/' + symbol_elle + '/X_valid.csv', index=False)
Y_valid.to_csv('data/mode sl/datasets std/' + symbol_elle + '/Y_valid.csv', index=False)

X_test.to_csv('data/mode sl/datasets std/' + symbol_elle + '/X_test.csv', index=False)
Y_test.to_csv('data/mode sl/datasets std/' + symbol_elle + '/Y_test.csv', index=False)
