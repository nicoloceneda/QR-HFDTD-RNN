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
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Serialize over all symbols

symbol_list = ['AAPL', 'AMD', 'AMZN', 'CSCO', 'FB', 'INTC', 'JPM', 'MSFT', 'NVDA', 'TSLA']

for symbol in symbol_list:

    print('Generating the dataset for:', symbol)

    # Import the extracted datasets

    data_extracted_ori = pd.read_csv('data/mode sl/datasets/' + symbol + '/data.csv')
    data_extracted_add = pd.read_csv('data/mode sl/datasets/' + symbol + '_addition/data.csv')

    # Create the features and target datasets

    log_return_ori = np.diff(np.log(data_extracted_ori['price']))
    data_ori = pd.DataFrame({'log_return': log_return_ori})

    log_return_add = np.diff(np.log(data_extracted_add['price']))
    data_add = pd.DataFrame({'log_return': log_return_add})

    date_change_ori = (data_extracted_ori['date'] != data_extracted_ori['date'].shift()).astype(int)
    date_change_ori = date_change_ori.iloc[1:].reset_index(drop=True)
    data_ori = data_ori[date_change_ori == 0].reset_index(drop=True)

    date_change_add = (data_extracted_add['date'] != data_extracted_add['date'].shift()).astype(int)
    date_change_add = date_change_add.iloc[1:].reset_index(drop=True)
    data_add = data_add[date_change_add == 0].reset_index(drop=True)

    elle = 200
    data_ori['log_return_ma'] = data_ori['log_return'].rolling(window=elle).mean()
    data_add['log_return_ma'] = np.NaN

    data = pd.concat([data_add, data_ori])

    data['log_return_mstd'] = data['log_return'].rolling(window=data_add.shape[0] + elle).std()

    Y = []
    X = []
    X2 = []

    for pos in range(data_add.shape[0] + elle, data.shape[0]):

        Y.append(data.iloc[pos]['log_return'])

        data_past = pd.DataFrame(data.iloc[pos - elle: pos], copy=True)
        r_past = data_past['log_return']
        r_past_ma = data_past.iloc[-1]['log_return_ma']
        r_diff = r_past - r_past_ma
        data_past['log_return_d2'] = r_diff ** 2
        data_past['log_return_d3'] = r_diff ** 3
        data_past['log_return_d4'] = r_diff ** 4
        X.append(data_past[['log_return', 'log_return_d2', 'log_return_d3', 'log_return_d4']])

        X2.append(data_past.iloc[-1]['log_return_mstd'])

    Y = pd.DataFrame(Y, columns=['label'])
    X = pd.concat(X, ignore_index=True)
    X2 = pd.DataFrame(X2, columns=['label'])

    # Define the training, validation and test subsets

    train_split = int(Y.shape[0] * 0.8)
    valid_split = train_split + int(Y.shape[0] * 0.1)
    test_split = valid_split + int(Y.shape[0] * 0.1)

    X_train = pd.DataFrame(X.iloc[:train_split * elle], copy=True)
    X2_train = pd.DataFrame(X2.iloc[:train_split], copy=True)
    Y_train = pd.DataFrame(Y.iloc[:train_split], copy=True)

    X_valid = pd.DataFrame(X.iloc[train_split * elle: valid_split * elle], copy=True)
    X2_valid = pd.DataFrame(X2.iloc[train_split: valid_split], copy=True)
    Y_valid = pd.DataFrame(Y.iloc[train_split: valid_split], copy=True)

    X_test = pd.DataFrame(X.iloc[valid_split * elle: test_split * elle], copy=True)
    X2_test = pd.DataFrame(X2.iloc[valid_split: test_split], copy=True)
    Y_test = pd.DataFrame(Y.iloc[valid_split: test_split], copy=True)

    # Standardize the training, validation and test subsets

    for column in X_train.columns:
        column_mean = X_train[column].mean()
        column_std = X_train[column].std()

        X_train[column] = (X_train[column] - column_mean) / column_std
        X_valid[column] = (X_valid[column] - column_mean) / column_std
        X_test[column] = (X_test[column] - column_mean) / column_std

    X2_mean = X2_train.mean()
    X2_std = X2_train.std()

    X2_train = (X2_train - X2_mean) / X2_std
    X2_valid = (X2_valid - X2_mean) / X2_std
    X2_test = (X2_test - X2_mean) / X2_std

    Y_mean = Y_train.mean()
    Y_std = Y_train.std()

    Y_train = (Y_train - Y_mean) / Y_std
    Y_valid = (Y_valid - Y_mean) / Y_std
    Y_test = (Y_test - Y_mean) / Y_std

    # Save the standardized returns

    if not os.path.isdir('data/mode sl/datasets std noj improved'):

        os.mkdir('data/mode sl/datasets std noj improved')

    symbol_elle = symbol + '_' + str(elle)

    if not os.path.isdir('data/mode sl/datasets std noj improved/' + symbol_elle):

        os.mkdir('data/mode sl/datasets std noj improved/' + symbol_elle)

    X_train.to_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/X_train.csv', index=False)
    X2_train.to_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/X2_train.csv', index=False)
    Y_train.to_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/Y_train.csv', index=False)

    X_valid.to_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/X_valid.csv', index=False)
    X2_valid.to_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/X2_valid.csv', index=False)
    Y_valid.to_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/Y_valid.csv', index=False)

    X_test.to_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/X_test.csv', index=False)
    X2_test.to_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/X2_test.csv', index=False)
    Y_test.to_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/Y_test.csv', index=False)
