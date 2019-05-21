""" data_analysis.py
    ----------------
    This script constructs the plot of the beginning and end of regular trading hours.

    Contact: nicolo.ceneda@student.unisg.ch
    Last update: 21 May 2019

"""


# ------------------------------------------------------------------------------------------------------------------------------------------
# PROGRAM SETUP
# ------------------------------------------------------------------------------------------------------------------------------------------

# Import the libraries and the modules

import matplotlib.pyplot as plt
import wrds


# Establish a connection to the wrds cloud

db = wrds.Connection()


# ------------------------------------------------------------------------------------------------------------------------------------------
# DATA EXTRACTION
# ------------------------------------------------------------------------------------------------------------------------------------------


# Run the SQL query

aapl_20190329_start = db.raw_sql("SELECT * FROM taqm_2019.ctm_20190329 WHERE sym_root = 'AAPL' and time_m >= '09:29:50' and time_m <= '09:30:10'")
aapl_20190329_end = db.raw_sql("SELECT * FROM taqm_2019.ctm_20190329 WHERE sym_root = 'AAPL' and time_m >= '15:59:50' and time_m <= '16:00:10'")


# ------------------------------------------------------------------------------------------------------------------------------------------
# DISPLAY THE PLOTS
# ------------------------------------------------------------------------------------------------------------------------------------------


# Create the subplots of the begin and end of the regular trading session

x1 = aapl_20190329_start.loc[:, 'time_m']
y1 = aapl_20190329_start.loc[:,'price']

x2 = aapl_20190329_end.loc[:, 'time_m']
y2 = aapl_20190329_end.loc[:, 'price']

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(32, 7))

ax[0].plot(x1, y1, label='AAPL, 2019-03-29, Opening', color='red', marker='+', markersize=2, linestyle='None')
ax[0].plot(x1.iloc[117], y1.iloc[117], color='black', marker='+', markersize=12)
ax[0].grid(color='dimgrey', linewidth=0.15)
ax[0].axvline(x='2019-03-29 09:30:00', linestyle=':', color='black', linewidth=2)
ax[0].set_xlim(x1.min(), x1.max())
ax[0].set_xlabel('time', fontsize=15)
ax[0].set_ylabel('price', fontsize=15)
ax[0].xaxis.set_tick_params(labelsize=15)
ax[0].yaxis.set_tick_params(labelsize=15)
ax[0].legend(fontsize=15, loc='lower left')
ax[0].xaxis.set_major_locator(plt.MaxNLocator(6))

ax[1].plot(x2, y2, label='AAPL, 2019-03-29, Closing', color='blue', marker='+', markersize=2, linestyle='None')
ax[1].plot(x2.iloc[1579], y2.iloc[1579], color='black', marker='+', markersize=12)
ax[1].grid(color='dimgrey', linewidth=0.15)
ax[1].axvline(x='2019-03-29 16:00:00', linestyle=':', color='black', linewidth=2)
ax[1].set_xlim(x2.min(), x2.max())
ax[1].set_xlabel('time', fontsize=15)
ax[1].set_ylabel('price', fontsize=15)
ax[1].xaxis.set_tick_params(labelsize=15)
ax[1].yaxis.set_tick_params(labelsize=15)
ax[1].legend(fontsize=15, loc='lower left')
ax[1].xaxis.set_major_locator(plt.MaxNLocator(4))

fig.tight_layout()
plt.savefig('z_Trading_Start_End.png')
