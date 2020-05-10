""" lstm_rnn.py
    -----------
    This script executes a long short term memory recurrent neural network to estimate the parameters of a parametric heavy tailed quantile
    function.

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
import matplotlib.pyplot as plt
import scipy.stats
import tensorflow as tf


# Set the parameters

symbol_list = ['AAPL', 'AMD', 'AMZN', 'CSCO', 'FB', 'INTC', 'JPM', 'MSFT', 'NVDA', 'TSLA']
run_list = [1, 2, 5, 6]
batch_size_list = [100, 4422, 100, 4422]
hidden_dim_list = [16, 16, 32, 32]
n_epochs_list = [10, 10, 10, 10]


# Iterate over each symbol

for symbol in symbol_list:

    for i in range(4):

        # Set the seed

        tf.random.set_seed(1)

        # Set the run

        run = run_list[i]
        print('Run {} for {}'.format(run, symbol))

        # -------------------------------------------------------------------------------
        # 1. PREPARE THE DATA
        # -------------------------------------------------------------------------------

        # Import the train, validation and test sets

        elle = 200

        symbol_elle = symbol + '_' + str(elle)

        X_train = pd.read_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/X_train.csv')
        X_valid = pd.read_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/X_valid.csv')
        X_test = pd.read_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/X_test.csv')

        Y_train = pd.read_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/Y_train.csv')
        Y_valid = pd.read_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/Y_valid.csv')
        Y_test = pd.read_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/Y_test.csv')

        X2_train = pd.read_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/X2_train.csv')
        X2_valid = pd.read_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/X2_valid.csv')
        X2_test = pd.read_csv('data/mode sl/datasets std noj improved/' + symbol_elle + '/X2_test.csv')

        # Reshape the train, validation and test subsets

        n_features = 4

        X_train = X_train.values.reshape(-1, elle, n_features)
        X_valid = X_valid.values.reshape(-1, elle, n_features)
        X_test = X_test.values.reshape(-1, elle, n_features)

        Y_train = Y_train.values
        Y_valid = Y_valid.values
        Y_test = Y_test.values

        X2_train = X2_train.values
        X2_valid = X2_valid.values
        X2_test = X2_test.values

        # Create train, validation and test tensorflow datasets

        batch_size = batch_size_list[i]
        """ PARAMS: 100, int(Y_valid.shape[0] / 10) """

        buffer_size = Y_train.shape[0]
        """ PARAMS: 10000 """

        ds_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        ds_train = ds_train.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        """ ALTERNATIVE: ds_train = ds_train.shuffle(10000).batch(batch_size).repeat() """

        ds_valid = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
        ds_valid = ds_valid.batch(batch_size, drop_remainder=True)
        """ ALTERNATIVE: ds_valid = ds_valid.batch(batch_size).repeat() """

        ds_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        ds_test = ds_test.batch(batch_size, drop_remainder=True)
        """ ALTERNATIVE: ds_test = ds_test.batch(batch_size).repeat() """

        for batch in ds_train.take(1):

            array_features = batch[0]
            array_targets = batch[1]

            print('The dataset is made up of several batches, each containing:'
                  '\n- An array of time series of features: shape=', array_features.shape,
                  '\n- An array of targets: shape=', array_targets.shape,
                  '\n'
                  '\nBatch 0'
                  '\n-------',
                  '\nTuple [0]\n',
                  array_features.numpy()[0], array_targets.numpy()[0],
                  '\nTuple [-1]\n',
                  array_features.numpy()[-1], array_targets.numpy()[-1])

        # -------------------------------------------------------------------------------
        # 2. DESIGN THE MODEL
        # -------------------------------------------------------------------------------

        # Create the model

        A = 4
        hidden_dim = hidden_dim_list[i]
        output_dim = 4

        lstm_model = tf.keras.models.Sequential()
        lstm_model.add(tf.keras.layers.LSTM(units=hidden_dim, return_sequences=False, input_shape=X_train.shape[-2:]))
        lstm_model.add(tf.keras.layers.Dense(units=output_dim, activation='tanh'))
        lstm_model.add(tf.keras.layers.Lambda(lambda x: x + np.array([0, 1, 1, 1])))

        # Print the model summary

        lstm_model.summary()

        print('\nLSTM layer'
              '\n----------'
              '\nWeight kernel:            4(FE x HU)          = (FE x 4HU) =', lstm_model.weights[0].shape,
              '\nWeight recurrent kernel:  4(HU x HU)          = (HU x 4HU) =', lstm_model.weights[1].shape,
              '\nWeight bias:              4(BS x HU) = 4(HU,) = (4HU,)     =', lstm_model.weights[2].shape,
              '\n                                                             -------'
              '\nTotal                                                          1344')

        print('\nDENSE layer'
              '\n-----------'
              '\nWeight kernel:                                              ', lstm_model.weights[3].shape,
              '\nWeight :                                                    ', lstm_model.weights[4].shape,
              '\n                                                             -------',
              '\nTotal                                                           68')

        # Create additional variables

        tau = tf.constant(np.concatenate(([0.01], np.divide(range(1, 20), 20), [0.99])).reshape(1, -1), dtype=tf.float32)
        z_tau = tf.constant(scipy.stats.norm.ppf(tau, loc=0.0, scale=1.0), dtype=tf.float32)

        # Define the loss function that produces a scalar for each batch (Equation 60)

        def q_calculator(params_predicted):                                         # (BS x 4)

            mu = tf.reshape(params_predicted[:, 0], [-1, 1])                        # (BS x 1)
            sig = tf.reshape(params_predicted[:, 1], [-1, 1])                       # (BS x 1)
            u_coeff = tf.reshape(params_predicted[:, 2], [-1, 1])                   # (BS x 1)
            d_coeff = tf.reshape(params_predicted[:, 3], [-1, 1])                   # (BS x 1)
            u_factor = tf.exp(tf.matmul(u_coeff, z_tau)) / A + 1               # (BS x 1)(1 x n_z_tau) = (BS x n_z_tau)
            d_factor = tf.exp(-tf.matmul(d_coeff, z_tau)) / A + 1              # (BS x 1)(1 x n_z_tau) = (BS x n_z_tau)
            prod_factor = tf.multiply(u_factor, d_factor)                      # (BS x n_z_tau)*(BS x n_z_tau) = (BS x n_z_tau)
            q = tf.add(mu, tf.multiply(sig, tf.multiply(z_tau, prod_factor)))  # (BS x 1)+(BS x 1)*(1 x n_z_tau)*(BS x n_z_tau) = (BS x n_z_tau)

            return q


        def pinball_loss_function(Y_actual, params_predicted):

            q = q_calculator(params_predicted)                                      # (BS x n_z_tau)
            error = tf.subtract(tf.cast(Y_actual, dtype=tf.float32), q)        # (BS x 1)-(BS x n_z_tau) = (BS x n_z_tau)
            error_1 = tf.multiply(tau, error)                                  # (1 x n_tau)*(BS x n_z_tau) = (BS x n_z_tau)
            error_2 = tf.multiply(tau - 1, error)                              # (1 x n_tau)*(BS x n_z_tau) = (BS x n_z_tau)
            loss = tf.reduce_mean(tf.maximum(error_1, error_2))                # (BS x n_z_tau) -> (1,)

            return loss


        # Compile the model

        lstm_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=pinball_loss_function)

        # -------------------------------------------------------------------------------
        # 3. TRAIN THE MODEL
        # -------------------------------------------------------------------------------

        # Train the lstm recurrent neural network

        n_epochs = n_epochs_list[i]

        history = lstm_model.fit(ds_train, epochs=n_epochs, validation_data=ds_valid)
        """ ALTERNATIVE: 
            n_steps_per_epoch = 600
            n_validation_steps = 100
           history = lstm_model.fit(ds_train, epochs=n_epochs, steps_per_epoch=n_steps_per_epoch,
                                    validation_data=ds_valid, validation_steps=n_validation_steps) 
        """

        # Visualize the learning curve

        if not os.path.isdir('data/mode sl/results noj/'):

            os.mkdir('data/mode sl/results noj/')

        if not os.path.isdir('data/mode sl/results noj/' + symbol_elle):

            os.mkdir('data/mode sl/results noj/' + symbol_elle)

        hist = history.history

        plt.figure()
        plt.plot(hist['loss'], 'b')
        plt.xlabel('Epoch')
        plt.title('Training loss')
        plt.tick_params(axis='both', which='major')
        plt.tight_layout()
        plt.savefig('data/mode sl/results noj/' + symbol_elle + '/train_loss_{}.png'.format(run))

        plt.figure()
        plt.plot(hist['val_loss'], 'r')
        plt.xlabel('Epoch')
        plt.title('Validation loss')
        plt.tick_params(axis='both', which='major')
        plt.tight_layout()
        plt.savefig('data/mode sl/results noj/' + symbol_elle + '/valid_loss_{}.png'.format(run))

        # -------------------------------------------------------------------------------
        # 4. MAKE PREDICTIONS
        # -------------------------------------------------------------------------------

        # Predict the parameters and the quantiles for the train and test subsets

        params_predicted_train = lstm_model.predict(X_train)
        params_predicted_train_df = pd.DataFrame(params_predicted_train, columns=['mu', 'sigma', 'u_coeff', 'd_coeff'])
        q_params_predicted_train = q_calculator(params_predicted_train)
        q_record_train_df = pd.DataFrame(q_params_predicted_train.numpy(), columns=tau.numpy().tolist()[0])

        params_predicted_valid = lstm_model.predict(X_valid)
        params_predicted_valid_df = pd.DataFrame(params_predicted_valid, columns=['mu', 'sigma', 'u_coeff', 'd_coeff'])
        q_params_predicted_valid = q_calculator(params_predicted_valid)
        q_record_valid_df = pd.DataFrame(q_params_predicted_valid.numpy(), columns=tau.numpy().tolist()[0])

        # -------------------------------------------------------------------------------
        # 4. MODEL IMPROVEMENT
        # -------------------------------------------------------------------------------

        # Set the seed

        tf.random.set_seed(1)

        # Reshape the train, validation and test subsets

        elle = 200
        n_features = 5

        X_train_new = pd.concat([params_predicted_train_df, X2_train], axis=1)
        X_train_new = X_train_new.values.reshape(-1, elle, n_features)

        X_valid_new = pd.concat([params_predicted_valid_df, X2_valid], axis=1)
        X_valid_new = X_valid_new.values.reshape(-1, elle, n_features)

        Y_train_new = Y_train
        Y_valid_new = Y_valid

        # Create train, validation and test tensorflow datasets

        batch_size = 100

        buffer_size = Y_train_new.shape[0]

        ds_train_new = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        ds_train_new = ds_train.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

        ds_valid_new = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
        ds_valid_new = ds_valid.batch(batch_size, drop_remainder=True)

        ds_test_new = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        ds_test_new = ds_test.batch(batch_size, drop_remainder=True)








        lstm_model = tf.keras.models.Sequential()
        lstm_model.add(tf.keras.layers.LSTM(units=hidden_dim, return_sequences=False, input_shape=X_train.shape[-2:]))
        lstm_model.add(tf.keras.layers.Dense(units=output_dim, activation='tanh'))
        lstm_model.add(tf.keras.layers.Lambda(lambda x: x + np.array([0, 1, 1, 1])))










        params_predicted_test = lstm_model.predict(X_test)
        params_predicted_test_df = pd.DataFrame(params_predicted_test, columns=['mu', 'sigma', 'u_coeff', 'd_coeff'])
        q_params_predicted_test = q_calculator(params_predicted_test)
        q_record_test_df = pd.DataFrame(q_params_predicted_test.numpy(), columns=tau.numpy().tolist()[0])

        # Save the predictions

        # params_predicted_train_df.to_csv('data/mode sl/results noj/' + symbol + '/params_predicted_train.csv', index=False)
        # q_record_train_df.to_csv('data/mode sl/results noj/' + symbol + '/q_params_predicted_train.csv', index=False)

        # params_predicted_test_df.to_csv('data/mode sl/results noj/' + symbol + '/params_predicted_test.csv', index=False)
        # q_record_test_df.to_csv('data/mode sl/results noj/' + symbol + '/q_params_predicted_test.csv', index=False)

        # Compute the test results noj

        loss_test_tau = pinball_loss_function(Y_test, params_predicted_test)

        tau = tf.constant(np.array([0.01, 0.05, 0.1]).reshape(1, -1), dtype=tf.float32)
        z_tau = tf.constant(scipy.stats.norm.ppf(tau, loc=0.0, scale=1.0), dtype=tf.float32)

        loss_test_new_tau = pinball_loss_function(Y_test, params_predicted_test)

        with open('data/mode sl/results noj/' + symbol_elle + '/results_{}.txt'.format(run), 'w') as file:

            file.write('LSTM RNN - Symbol: {}'.format(symbol))
            file.write('\n- Sequence length: {}'.format(elle))
            file.write('\n- Batch size: {}'.format(batch_size))
            file.write('\n- Hidden dimension: {}'.format(hidden_dim))
            file.write('\n- Number of epochs: {}'.format(n_epochs))
            file.write('\n- Number of steps per epochs: {}'.format(n_steps_per_epoch))
            file.write('\n- Number of steps per validation: {}'.format(n_validation_steps))
            file.write('\n* Test loss (tau): {}'.format(loss_test_tau))
            file.write('\n* Test loss (new tau): {}'.format(loss_test_new_tau))
            file.write('\n')
            file.write('\nTrain loss: \n{}'.format(hist['loss']))
            file.write('\n')
            file.write('\nValid loss: \n{}'.format(hist['val_loss']))


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# plt.show()


