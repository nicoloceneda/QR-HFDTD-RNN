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

def graph_output(output_, symbol_list_, date_index_, usage_):

    date_grid, symbol_grid = np.meshgrid(date_index_, symbol_list_)
    date_symbol_grid = np.array([date_grid.ravel(), symbol_grid.ravel()]).T

    fig = plt.figure(figsize=(20,10))
    h = len(date_index_)
    w = len(symbol_list_)
    pos = 0

    for date, symbol in date_symbol_grid:

        pos += 1
        x = output_.loc[(output_.loc[:, 'sym_root'] == symbol) & (pd.to_datetime(output_.loc[:, 'date']) == pd.to_datetime(date)), 'time_m']
        y = output_.loc[(output_.loc[:, 'sym_root'] == symbol) & (pd.to_datetime(output_.loc[:, 'date']) == pd.to_datetime(date)), 'price']

        ax = fig.add_subplot(h, w, pos)
        ax.plot(x, y, linewidth=0.1)
        ax.set_title('{} {} {}'.format(symbol, str(pd.to_datetime(date))[:10], usage_))

    fig.tight_layout()
    plt.savefig('z_{}.png'.format(usage_))


# Create a function to display comparative plots for the same symbol and date but different output status

def graph_comparison(output1_, output2_, symbol_, date_, usage1_, usage2_):

    x1 = output1_.loc[(output1_.loc[:, 'sym_root'] == symbol_) & (pd.to_datetime(output1_.loc[:, 'date']) == pd.to_datetime(date_)), 'time_m']
    y1 = output1_.loc[(output1_.loc[:, 'sym_root'] == symbol_) & (pd.to_datetime(output1_.loc[:, 'date']) == pd.to_datetime(date_)), 'price']
    label1 = symbol_ + ', ' + str(pd.to_datetime(date_))[:10] + ', ' + usage1_

    x2 = output2_.loc[(output2_.loc[:, 'sym_root'] == symbol_) & (pd.to_datetime(output2_.loc[:, 'date']) == pd.to_datetime(date_)), 'time_m']
    y2 = output2_.loc[(output2_.loc[:, 'sym_root'] == symbol_) & (pd.to_datetime(output2_.loc[:, 'date']) == pd.to_datetime(date_)), 'price']
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


# Create a function to print the output dataframes

def print_output(output_, print_output_flag_, head_flag_):

    if (print_output_flag_ is True) & (head_flag_ is True):

        print(output_.head())

    elif (print_output_flag_ is True) & (head_flag_ is False):

        print(output_)

    else:

        print('"Print output" is not active')
