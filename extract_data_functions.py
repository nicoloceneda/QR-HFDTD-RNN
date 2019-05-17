""" extract_data_functions.py
    -------------------------
    This script contains general functions called in 'extract_data.py'. Functions specific to the 'extract_data.py' are not contained in this
    script.

    Contact: nicolo.ceneda@student.unisg.ch
    Last update: 17 May 2019

"""


# ------------------------------------------------------------------------------------------------------------------------------------------
# PROGRAM SETUP
# ------------------------------------------------------------------------------------------------------------------------------------------


# Import the libraries and the modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------------------------------------------------------------------


# Create a function to separate the sections in the output file

def section(my_string):

    print()
    print('-----------------------------------------------------------------------------------------------')
    print('{}'.format(my_string))
    print('-----------------------------------------------------------------------------------------------')


# Create a function to display the plots of the specified symbols and dates

def graph_output(output_, symbol_list_, date_index_, usage):

    date_grid, symbol_grid = np.meshgrid(date_index_, symbol_list_)
    date_symbol_grid = np.array([date_grid.ravel(), symbol_grid.ravel()]).T

    fig = plt.figure(figsize=(20,10))
    h = len(date_index_)
    w = len(symbol_list_)
    pos = 0

    for date, symbol in date_symbol_grid:

        pos += 1
        x = output_.loc[(output_.loc[:, 'sym_root'] == str(symbol)) & (pd.to_datetime(output_.loc[:, 'date']) == pd.to_datetime(date)), 'time_m']
        y = output_.loc[(output_.loc[:, 'sym_root'] == symbol) & (pd.to_datetime(output_.loc[:, 'date']) == pd.to_datetime(date)), 'price']

        ax = fig.add_subplot(h, w, pos)
        ax.plot(x, y, linewidth=0.1)
        ax.set_title('{} {} {}'.format(symbol, str(pd.to_datetime(date))[:10], usage))

    plt.savefig('z_{}.png'.format(usage))


# Create a function to reset the index

def reset_index(output_):

    output_ = output_.set_index(pd.Index(range(output_.shape[0]), dtype='int64'))

    return output_
