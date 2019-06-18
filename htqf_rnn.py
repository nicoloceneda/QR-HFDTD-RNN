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
import tensorflow as tf
from datetime import datetime


# Create a function to

# TODO: generalize all the parameters
# TODO: extend the class to run also the data_extract process

class HtqfRnn(object):

    def __init__(self, symbol, num_batches, elle, n_features, hidden_layer_dim, output_layer_dim, num_layers, epochs, learning_rate, a):

        self.symbol = symbol
        self.num_batches = num_batches
        self.elle = elle
        self.n_features = n_features
        self.hidden_layer_dim = hidden_layer_dim
        self.output_layer_dim = output_layer_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.a = a

        # Build the computational graph

        tf.set_random_seed(123)

        self.g = tf.Graph()

        with self.g.as_default():

            # Placeholders, variables constants

            X_tf = tf.placeholder(dtype=tf.float32, shape=(None, self.elle, self.n_features), name='X_tf')
            Y_tf = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='Y_tf')
            tau_tf = tf.placeholder(dtype=tf.float32, shape=(1, None), name='tau_tf')
            z_tf = tf.placeholder(dtype=tf.float32, shape=(1, None), name='z_tf')

            w_out_tf = tf.Variable(tf.random_normal(shape=(self.hidden_layer_dim, self.output_layer_dim)), dtype=tf.float32)
            b_out_tf = tf.Variable(tf.random_normal(shape=(1, self.output_layer_dim)), dtype=tf.float32)

            self.init = tf.global_variables_initializer()

            add_constant_tf = tf.constant([0, 1, 1, 1], dtype=tf.float32, shape=(1, 4), name='add_constant_tf')                             # TODO: inq

            # LSTM RNN

            cells_one_layer = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_layer_dim, name='cells_one_layer')                              # TODO: use cuda for GPu performance
            outputs_lstm, state_lstm, = tf.nn.dynamic_rnn(cell=cells_one_layer, inputs=X_tf, dtype=tf.float32)                              # TODO add more layers
                                                                                                                                            # TODO: use keras.layers.RNN
            # Output

            out = tf.tanh(tf.add(tf.matmul(state_lstm.h, w_out_tf), b_out_tf))
            parameters = tf.add(out, add_constant_tf)

            # HTQF

            mu = tf.reshape(tensor=parameters[:, 0], shape=(-1, 1))
            sigma = tf.reshape(tensor=parameters[:, 1], shape=(-1, 1))
            tail_up = tf.reshape(tensor=parameters[:, 2], shape=(-1, 1))
            tail_down = tf.reshape(tensor=parameters[:, 3], shape=(-1, 1))

            factor_up = tf.add(tf.divide(tf.exp(tf.matmul(tail_up, z_tf)), self.a), 1)                                                      # TODO: check that it exactly does / A + 1
            factor_down = tf.add(tf.divide(tf.exp(-tf.matmul(tail_down, z_tf)), self.a), 1)
            factor_mult = tf.multiply(factor_up, factor_down)
            quantile = tf.add(mu, tf.multiply(tf.multiply(z_tf, factor_mult), sigma))

            # Error

            error = tf.subtract(Y_tf - quantile)
            pinball_1 = tf.multiply(tau_tf, error)
            pinball_2 = tf.multiply(tau_tf - 1, error)
            loss = tf.reduce_mean(tf.maximum(pinball_1, pinball_2))                                                                         # TODO: inq max

            # Optimization

            optimizer = tf.train.AdamOptimizer(self.learning_rate)                                                                          # TODO: inq AdamOptimizer
            train = optimizer.minimize(loss)


    def train(self):

        data_folder = 'bg/{}/{}/'.format(self.symbol, self.elle)

        X_train = pd.read_csv(data_folder + 'X_train.csv').values.reshape(-1, self.elle, self.n_features)
        X_valid = pd.read_csv(data_folder + 'X_valid.csv').values.reshape(-1, self.elle, self.n_features)

        Y_train = pd.read_csv(data_folder + 'Y_train.csv').values
        Y_valid = pd.read_csv(data_folder + 'Y_valid.csv').values

        with tf.Session(graph=self.g) as sess:

            sess.run(self.init)
            np.random.seed(0)

            for var in tf.trainable_variables():

                var.load(0.1 * np.random.randn(*sess.run(tf.shape(var))))                                                                   # TODO: inq: load

            start_time = datetime.now()
            max_valid_date = 1e8
            record_variables = []
            record_valid_date = []

            batch_list = self.allocate_batch(Y_train)

            pos = 0

            for step in range(10001):                                                                                                       # TODO: generalize

                feed = {'X:0': X_train[batch_list[pos], :, :], 'Y:0': Y_train[batch_list[pos]], 'tau:0': self.tau_long,
                        'percent:0': self.tau_long_percent, 'keep_prob:0': 0.5}
                loss, _ = sess.run(['loss:0', 'train'], feed_dict=feed)                                                                     # TODO: generalize keep proba




AAPL = HtqfRnn(symbol='AAPL', elle=100, n_features=4, hidden_layer_dim=16, output_layer_dim=4, num_layers=1, epochs=5, learning_rate=0.001, a=4)

# Import the training, validation and test data

X_test = pd.read_csv(data_folder + 'X_test.csv').values.reshape(-1, self.elle, self.n_features)


Y_test = pd.read_csv(data_folder + 'Y_test.csv').values

# Sets of tau probability levels and corresponding percentiles for training and test

self.tau_long = np.concatenate(([0.01], np.arange(0.05, 1, 0.05), [0.99])).reshape(1, -1)
self.tau_long_percent = scipy.stats.norm.ppf(self.tau_long)

self.tau_short = np.array([0.01, 0.05, 0.10]).reshape(1, -1)
self.tau_short_percent = scipy.stats.norm.ppf(self.tau_short).reshape(1, -1)



        for step in range(10001):

            sess.run(train, feed_dict={X: X_train[batch_list[ibatch], :, :], Y: Y_train[batch_list[ibatch]], tau: tau_full, perc: tau_full_perc})
            ibatch = ibatch + 1 if ibatch + 1 < epochs else 0
            record_variables.append([sess.run(variable) for variable in tf.trainable_variables()])
            record_valid_date.append(sess.run(loss, feed_dict={X: X_valid, Y: Y_valid, tau: tau_full, perc: tau_full_perc}))

            if step % 100 == 0:

                print(step, sess.run(loss, feed_dict={X: X_train, Y: Y_train, tau: tau_full, perc: tau_full_perc}), record_valid_date[-1])

                if record_valid_date[-1] > max_valid_date:

                    min_I = np.argmin(record_valid_date)

                    for variable, value in zip(tf.trainable_variables(), record_variables[minI]):

                        variable.load(value)

                    break

                else:

                    max_valid_date = record_valid_date[-1]
                    record_variables = [record_variables[-1]]
                    record_valid_date = [record_valid_date[-1]]

        Q_record_train = sess.run(Q, feed_dict={X: X_train, Y: Y_train, tau: tau_full, perc: tau_full_perc})
        params_record_train = sess.run(params, feed_dict={X: X_train, Y: Y_train, tau: tau_full, perc: tau_full_perc})
        Q_test_train = sess.run(Q, feed_dict={X: X_test, Y: Y_test, tau: tau_full, perc: tau_full_perc})
        params_record_test = sess.run(params, feed_dict={X: X_test, Y: Y_test, tau: tau_full, perc: tau_full_perc})

        result_folder = '{}/LSTM-HTQF_{}_{}/'.format(symbol, l, hidden_layer_dim)

        if not os.path.isdir(result_folder):

            os.mkdir(result_folder)

        pd.DataFrame(Q_record_train, columns=perc.tolist()[0]).to_csv(result_folder + "Qrecord_train.csv", index=False)
        pd.DataFrame(params_record_train, columns=['mu', 'sigma', 'u', 'v']).to_csv(result_folder + "paramsRecord_train.csv", index=False)
        pd.DataFrame(Q_test_train, columns=perc.tolist()[0]).to_csv(result_folder + "Qrecord_test.csv", index=False)
        pd.DataFrame(params_record_test, columns=['mu', 'sigma', 'u', 'v']).to_csv(result_folder + "paramsRecord_test.csv", index=False)

        with open(result_folder + "print.txt", 'w') as file:

            file.write("\nTime Cost: {:.2f} Minutes\n".format((datetime.now() - start_time).total_seconds() / 60.))
            file.write("My Validate Performance: {}\n".format(sess.run(loss, feed_dict={X: X_valid, Y: Y_valid, tau: tau_full, perc: tau_full_perc})))

            file.write("My Test Performance: {}\n".format(sess.run(loss, feed_dict={X: X_test, Y: Y_test, tau: tau_full, perc: tau_full_perc})))
            file.write("New Test Performance: {}\n".format(sess.run(loss, feed_dict={X: X_test, Y: Y_test, tau: tau_new, perc: tau_new_perc})))
            file.write("Sequence Length: {}, Hidden Dimension: {}\n\n".format(l, hidden_layer_dim))
            file.write("================================================================\n")




































