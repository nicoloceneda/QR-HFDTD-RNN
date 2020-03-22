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

    line_length = len(my_string)
    print('\n', '-'*line_length, '\n', '{}'.format(my_string), '\n', '-'*line_length)


# Create a function to print the output dataframes

def print_output(output_, print_output_flag_, head_flag_):

    if (print_output_flag_ is True) & (head_flag_ is True):

        print(output_.head(10))

    elif (print_output_flag_ is True) & (head_flag_ is False):

        print(output_)

    else:

        print('"Print output" is not active')


# Create a function to display the plots of the specified symbols and dates

def graph_output(output_, symbol_list_, date_index_, usage_):
    date_grid, symbol_grid = np.meshgrid(date_index_, symbol_list_)
    date_symbol = np.array([date_grid.ravel(), symbol_grid.ravel()]).T

    h = len(date_index_)
    w = len(symbol_list_)

    fig, ax = plt.subplots(nrows=h, ncols=w, sharex=True, figsize=(20, 10))
    ax = ax.flatten()

    for i in range(len(date_symbol)):
        symbol = date_symbol[i, 1]
        date = pd.to_datetime(date_symbol[i, 0])
        condition = (output_['sym_root'] == symbol) & (pd.to_datetime(output_['date']) == date)
        x = output_.loc[condition, 'time_m']
        y = output_.loc[condition, 'price']
        ax[i].plot(x, y, linewidth=0.1)
        ax[i].set_title('{} {} {}'.format(symbol, str(pd.to_datetime(date))[:10], usage_))

    fig.tight_layout()
    plt.savefig('z_{}.png'.format(usage_))


# Create a function to display comparative plots for the same symbol and date but different output status

def graph_comparison(output1_, output2_, symbol_, date_, usage1_, usage2_):

    condition_1 = (output1_['sym_root'] == symbol_) & (pd.to_datetime(output1_['date']) == pd.to_datetime(date_))
    x1 = output1_.loc[condition_1, 'time_m']
    y1 = output1_.loc[condition_1, 'price']
    label1 = symbol_ + ', ' + str(pd.to_datetime(date_))[:10] + ', ' + usage1_

    condition_2 = (output2_['sym_root'] == symbol_) & (pd.to_datetime(output2_['date']) == pd.to_datetime(date_))
    x2 = output2_.loc[condition_2, 'time_m']
    y2 = output2_.loc[condition_2, 'price']
    label2 = symbol_ + ', ' + str(pd.to_datetime(date_))[:10] + ', ' + usage2_

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(30, 10))

    ax[0].plot(x1, y1, label=label1, color='red', linewidth=0.3)
    ax[0].grid(color='dimgrey', linewidth=0.15)
    ax[0].set_xlim(x1.min(), x1.max())
    ax[0].set_xlabel('time', fontsize=25)
    ax[0].set_ylabel('price', fontsize=25)
    ax[0].xaxis.set_tick_params(labelsize=25)
    ax[0].yaxis.set_tick_params(labelsize=25)
    ax[0].legend(fontsize=25)

    ax[1].plot(x2, y2, label=label2, color='blue', linewidth=0.3)
    ax[1].grid(color='dimgrey', linewidth=0.15)
    ax[1].set_xlim(x2.min(), x2.max())
    ax[1].set_xlabel('time', fontsize=25)
    ax[1].set_ylabel('price', fontsize=25)
    ax[1].xaxis.set_tick_params(labelsize=25)
    ax[1].yaxis.set_tick_params(labelsize=25)
    ax[1].legend(fontsize=25)

    fig.tight_layout()
    plt.savefig('z_{}_{}.png'.format(usage1_, usage2_))



