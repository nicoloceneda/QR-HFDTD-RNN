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

import numpy as np
import pandas as pd
import scipy.stats
import tensorflow as tf
from datetime import datetime
import os


# Create the HtqfRnn class

# TODO: generalize all the parameters
# TODO: extend the class to run also the data_extract process

class HtqfRnn(object):

    # Constructor

    def __init__(self, symbol, elle, n_features, hidden_layer_dim, output_layer_dim, num_layers, epochs, learning_rate, a):

        # Initialize class variables

        self.symbol = symbol
        self.elle = elle
        self.n_features = n_features
        self.hidden_layer_dim = hidden_layer_dim
        self.output_layer_dim = output_layer_dim
        self.num_layers = num_layers
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.a = a

        # Import training, validation and test sets

        data_folder = 'sl/{}/{}/'.format(self.symbol, self.elle)

        self.X_train = pd.read_csv(data_folder + 'X_train.csv').values.reshape(-1, self.elle, self.n_features)
        self.Y_train = pd.read_csv(data_folder + 'Y_train.csv').values

        self.X_valid = pd.read_csv(data_folder + 'X_valid.csv').values.reshape(-1, self.elle, self.n_features)
        self.Y_valid = pd.read_csv(data_folder + 'Y_valid.csv').values

        self.X_test = pd.read_csv(data_folder + 'X_test.csv').values.reshape(-1, self.elle, self.n_features)
        self.Y_test = pd.read_csv(data_folder + 'Y_test.csv').values

        # Generate probability levels and corresponding quantiles for a standard normal

        self.tau_long = np.concatenate(([0.01], np.arange(0.05, 1, 0.05), [0.99])).reshape(1, -1)
        self.z_long = scipy.stats.norm.ppf(self.tau_long)

        self.tau_short = np.array([0.01, 0.05, 0.10]).reshape(1, -1)
        self.z_short = scipy.stats.norm.ppf(self.tau_short).reshape(1, -1)

        # Build the computational graph

        self.g = tf.Graph()

        with self.g.as_default():

            tf.set_random_seed(123)
            self.saver = tf.train.Saver()

            # Placeholders, variables constants

            X_tf = tf.placeholder(dtype=tf.float32, shape=(None, self.elle, self.n_features), name='X_tf')
            Y_tf = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='Y_tf')
            tau_tf = tf.placeholder(dtype=tf.float32, shape=(1, None), name='tau_tf')
            z_tf = tf.placeholder(dtype=tf.float32, shape=(1, None), name='z_tf')
            keep_prob_tf = tf.placeholder(dtype=tf.float32, name='tf_prob')

            w_out_tf = tf.Variable(tf.random_normal(shape=(self.hidden_layer_dim, self.output_layer_dim)), dtype=tf.float32)
            b_out_tf = tf.Variable(tf.random_normal(shape=(1, self.output_layer_dim)), dtype=tf.float32)

            self.init_op = tf.global_variables_initializer()

            add_constant_tf = tf.constant([0, 1, 1, 1], dtype=tf.float32, shape=(1, 4), name='add_constant_tf')                             # TODO: inq

            # LSTM RNN

            cells_one_layer = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_layer_dim, name='cells_one_layer')                              # TODO: use cuda for GPu performance
            cells_drop_out = tf.nn.rnn_cell.DropoutWrapper(cell=cells_one_layer, output_keep_prob=keep_prob_tf)
            cells_all_layer = [cells_drop_out for i in range(self.num_layers)]
            cells = tf.contrib.rnn.MultiRNNCell(cells=cells_all_layer)                                                                      # TODO: initial state

            self.initial_state = cells.zero_state(batch_size=1, dtype=tf.float32)

            outputs_lstm, self.state_lstm, = tf.nn.dynamic_rnn(cell=cells, inputs=X_tf, dtype=tf.float32)                                        # TODO add more layers
                                                                                                                                            # TODO: use keras.layers.RNN
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

            error = tf.subtract(Y_tf, quantile)
            pinball_1 = tf.multiply(tau_tf, error)
            pinball_2 = tf.multiply(tau_tf - 1, error)
            cost = tf.reduce_mean(tf.maximum(pinball_1, pinball_2))                                                                         # TODO: inq max

            # Optimization

            optimizer = tf.train.AdamOptimizer(self.learning_rate)                                                                          # TODO: inq AdamOptimizer
            train_op = optimizer.minimize(cost, name='train_op')

    # Train method

    def train(self):

        with tf.Session(graph=self.g) as sess:

            sess.run(self.init_op)
            iteration = 1

            for epoch in range(self.n_epochs):

                state = sess.run(self.initial_state)

                for batch in range(self.Y_train.shape[0]):

                    feed = {'X_tf:0': self.X_train[batch, :, :], 'Y_tf:0': self.Y_train[batch], 'tau_tf:0': self.tau_long, 'z_tf:0': self.z_long, 'keep_prob_tf:0': 0.5, self.initial_state: state}
                    loss, _, state = sess.run(['cost:0', 'train_op', self.state_lstm], feed_dict=feed)

                    if iteration % 100 == 0:

                        print("Epoch: {}/{} | Iteration: {} | Train loss: {:.5f}".format(epoch + 1, self.n_epochs, iteration, loss))

                    iteration += 1

                if (epoch + 1) % 10 == 0:

                    self.saver.save(sess, "model/{}_{}.ckpt".format(symbol, epoch))

    def predict(self):

        predictions = []

        with tf.Session(graph=self.g) as sess:

            self.saver.restore(sess, tf.train.latest_checkpoint('model/'))
            test_state = sess.run(self.initial_state)

            for batch in range(self.Y_train.shape[0]):

                feed = {'X_tf:0': X_train[batch, :, :], 'tau_tf:0': self.tau_long, 'z_tf:0': self.tau_long_q, 'keep_prob_tf:0': 1.0, self.initial_state: test_state}

                if return_probability:

                    pred, test_state = sess.run(['probabilities:0', self.final_state], feed_dict=feed)

                else:

                    pred, test_state = sess.run(['labels:0', self.final_state], feed_dict=feed)

                preds.append(pred)

        return np.concatenate(preds)


            np.random.seed(0)

            for var in tf.trainable_variables():

                var.load(0.1 * np.random.randn(*sess.run(tf.shape(var))))                                                                   # TODO: inq: load

            self.start_time = datetime.now()
            last_valid_loss = 1e8
            record_variables = []
            record_valid_loss = []

            pos = 0

            for batch in range(self.Y_train.shape[0]):                                                                                      # TODO: generalize

                feed_dict_train_batch = {'X:0': self.X_train[batch, :, :], 'Y:0': self.Y_train[batch], 'tau_tf:0': self.tau_long, 'z_tf:0': self.tau_long_q} # TODO: generalize keep proba
                sess.run('train', feed_dict=feed_dict_train_batch)

                feed_dict_valid_set = {'X:0': self.X_valid, 'Y:0': self.Y_valid, 'tau_tf:0': self.tau_long, 'z_tf:0': self.tau_long_q}
                record_valid_loss.append(sess.run('loss:0', feed_dict=feed_dict_valid_set))

                record_variables.append([sess.run(var) for var in tf.trainable_variables()])

                if batch % 100 == 0:

                    feed_dict_train_set = {'X:0': self.X_train, 'Y:0': self.Y_train, 'tau_tf:0': self.tau_long, 'z_tf:0': self.tau_long_q}
                    print('\nbatch: ', batch, ' | Loss training set: ', sess.run('loss:0', feed_dict=feed_dict_train_set), ' | Loss validation set', record_valid_loss[-1])

                    if record_valid_loss[-1] > last_valid_loss:

                        min_loss = np.argmin(last_valid_loss)

                        for var, value in zip(tf.trainable_variables(), record_variables[min_loss]):

                            var.load(value)

                        break

                    else:

                        last_valid_loss = record_valid_loss[-1]
                        record_valid_loss = [last_valid_loss]
                        record_variables = [record_variables[-1]]

    def predict(self):

        with tf.Session(graph=self.g) as sess:

            q_record_train = sess.run('quantile:0', feed_dict={'X:0': self.X_train, 'Y:0': self.Y_train, 'tau_tf:0': self.tau_long, 'z_tf:0': self.tau_long_q})
            parameters_record_train = sess.run('parameters', feed_dict={'X:0': self.X_train, 'Y:0': self.Y_train, 'tau_tf:0': self.tau_long, 'z_tf:0': self.tau_long_q})

            q_record_test = sess.run('quantile:0', feed_dict={'X:0': self.X_test, 'Y:0': self.Y_test, 'tau_tf:0': self.tau_long, 'z_tf:0': self.tau_long_q})
            parameters_record_test = sess.run('parameters', feed_dict={'X:0': self.X_test, 'Y:0': self.Y_test, 'tau_tf:0': self.tau_long, 'z_tf:0': self.tau_long_q})

            result_folder = '{}/LSTM-HTQF_{}_{}/'.format(self.symbol, self.elle, self.hidden_layer_dim)

            if not os.path.isdir(result_folder):

                os.mkdir(result_folder)

            pd.DataFrame(q_record_train, columns=self.tau_long.tolist()[0]).to_csv(result_folder + "q_record_train.csv", index=False)
            pd.DataFrame(parameters_record_train, columns=['mu', 'sigma', 'u', 'v']).to_csv(result_folder + "parameters_record_train.csv", index=False)
            pd.DataFrame(q_record_test, columns=self.tau_long.tolist()[0]).to_csv(result_folder + "q_record_test.csv", index=False)
            pd.DataFrame(parameters_record_test, columns=['mu', 'sigma', 'u', 'v']).to_csv(result_folder + "parameters_record_test.csv", index=False)

            with open(result_folder + "print.txt", 'w') as file:

                file.write("\nTime Cost: {:.2f} Minutes\n".format((datetime.now() - self.start_time).total_seconds() / 60.))
                file.write("My Validate Performance: {}\n".format(sess.run('loss:0', feed_dict={'X:0': self.X_valid, 'Y:0': self.Y_valid, 'tau_tf:0': self.tau_long, 'z_tf:0': self.tau_long_q})))
                file.write("My Test Performance: {}\n".format(sess.run('loss:0', feed_dict={'X:0': self.X_test, 'Y:0': self.Y_test, 'tau_tf:0': self.tau_long, 'z_tf:0': self.tau_long_q})))
                file.write("New Test Performance: {}\n".format(sess.run('loss:0', feed_dict={'X:0': self.X_test, 'Y:0': self.Y_test, 'tau_tf:0': self.tau_short, 'z_tf:0': self.tau_short_q})))
                file.write("Sequence Length: {}, Hidden Dimension: {}\n\n".format(self.elle, self.hidden_layer_dim))
                file.write("================================================================\n")

AAPL = HtqfRnn(symbol='AAPL', elle=100, n_features=4, hidden_layer_dim=16, output_layer_dim=4, num_layers=1, epochs=5, learning_rate=0.001, a=4)


self.saver = tf.train.Saver()
















































