""" IMAGES DOCUMENT
    ---------------
    This script generates some of the illustrations used in the paper.

    Contact: nicolo.ceneda@student.unisg.ch
    Last update: 18 May 2020
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import wrds


# Create the directory to store the images:

if not os.path.isdir('images_document'):

    os.mkdir('images_document')


# -------------------------------------------------------------------------------
# 4.2. DATA: SAMPLES
# -------------------------------------------------------------------------------

db = wrds.Connection()

opening_trades = db.raw_sql("SELECT date, time_m, sym_root, sym_suffix, tr_scond, size, price, tr_corr "
                            "FROM taqm_2019.ctm_20190329 "
                            "WHERE sym_root = 'AAPL' "
                            "AND sym_suffix is null "
                            "AND time_m >= '09:29:50' "
                            "AND time_m <= '09:30:10' ")

index_resample = pd.DatetimeIndex(opening_trades['date'].apply(str) + ' ' + opening_trades['time_m'].apply(str))
opening_trades = opening_trades.set_index(index_resample)
opening_trade = opening_trades[opening_trades['tr_scond'] == '@  Q'][:1]

closing_trades = db.raw_sql("SELECT date, time_m, sym_root, sym_suffix, tr_scond, size, price, tr_corr "
                            "FROM taqm_2019.ctm_20190329 "
                            "WHERE sym_root = 'AAPL' "
                            "AND sym_suffix is null "
                            "AND time_m >= '15:59:50' "
                            "AND time_m <= '16:00:10' ")

index_resample = pd.DatetimeIndex(closing_trades['date'].apply(str) + ' ' + closing_trades['time_m'].apply(str))
closing_trades = closing_trades.set_index(index_resample)
closing_trade = closing_trades[closing_trades['tr_scond'] == '@  M'][:1]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))

ax[0].scatter(opening_trades.index, opening_trades['price'],  marker='+', s=20, color='blue', label='AAPL, 2019-03-29, Open')
ax[0].scatter(opening_trade.index, opening_trade['price'],  marker='s', s=40, edgecolor='k', facecolors='none')
ax[0].axvline(pd.to_datetime('2019-03-29 09:30:00.000000'), linestyle='-', linewidth=0.5, color='k')
ax[0].grid(linewidth=0.3)
ax[0].set_xlabel('Time', fontsize=14)
ax[0].set_ylabel('Price', fontsize=14)
ax[0].tick_params(axis='x', labelsize=12)
ax[0].tick_params(axis='y', labelsize=12)
ax[0].legend(fontsize=10, loc='lower left')

ax[1].scatter(closing_trades.index, closing_trades['price'],  marker='+', s=20, color='red', label='AAPL, 2019-03-29, Close')
ax[1].scatter(closing_trade.index, closing_trade['price'],  marker='s', s=40, edgecolor='k', facecolors='none')
ax[1].axvline(pd.to_datetime('2019-03-29 16:00:00.000000'), linestyle='-', linewidth=0.5, color='k')
ax[1].grid(linewidth=0.3)
ax[1].set_xlabel('Time', fontsize=14)
ax[1].set_ylabel('Price', fontsize=14)
ax[1].tick_params(axis='x', labelsize=12)
ax[1].tick_params(axis='y', labelsize=12)
ax[1].legend(fontsize=10, loc='lower left')

fig.tight_layout()
plt.savefig('images_document/z_opening_close.png')

# -------------------------------------------------------------------------------
# 5.3. MODEL: Q-Q PLOTS
# -------------------------------------------------------------------------------


# Define the heavy tail quantile function

def htqf(mu, sigma, u, d, A):

    return mu + sigma * qantile_std_normal * (np.exp(u * qantile_std_normal) / A + 1) * (np.exp(-d * qantile_std_normal) / A + 1)


# Plot the QQ plots

mu = 1
sigma = 1.5
tau = np.arange(0.001, 1, 0.001)

qantile_std_normal = scipy.stats.norm.ppf(tau, loc=0.0, scale=1)
qantile_normal = scipy.stats.norm.ppf(tau, loc=mu, scale=sigma)
qantile_student = scipy.stats.t.ppf(tau, 2, loc=mu, scale=sigma)
htqf_1 = htqf(mu=mu, sigma=sigma, u=1.0, d=0.1, A=4)
htqf_2 = htqf(mu=mu, sigma=sigma, u=0.6, d=1.2, A=4)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

ax[0].plot(qantile_std_normal, qantile_normal, linestyle='--', color='blue', label='N(1, 1.5)')
ax[0].plot(qantile_std_normal, qantile_student, linestyle='-', color='red', label='t(2)')
ax[0].axhline(0.0, linestyle='--', linewidth=0.5, color='k')
ax[0].axvline(0.0, linestyle='--', linewidth=0.5, color='k')
ax[0].set_xlim(-3, 3)
ax[0].set_ylim(-12, 12)
ax[0].set_xlabel('N(0,1)', fontsize=14)
ax[0].tick_params(axis='x', labelsize=12)
ax[0].tick_params(axis='y', labelsize=12)
ax[0].legend(fontsize=10)

ax[1].plot(qantile_std_normal, qantile_normal, linestyle='--', color='blue', label='N(1, 1.5)')
ax[1].plot(qantile_std_normal, htqf_1, linestyle='-', color='red', label='htqf: u=1.0, d=0.1')
ax[1].axhline(0.0, linestyle='--', linewidth=0.5, color='k')
ax[1].axvline(0.0, linestyle='--', linewidth=0.5, color='k')
ax[1].set_xlim(-3, 3)
ax[1].set_ylim(-12, 12)
ax[1].set_xlabel('N(0,1)', fontsize=14)
ax[1].tick_params(axis='x', labelsize=12)
ax[1].tick_params(axis='y', labelsize=12)
ax[1].legend(fontsize=10)

ax[2].plot(qantile_std_normal, qantile_normal, linestyle='--', color='blue', label='N(1, 1.5)')
ax[2].plot(qantile_std_normal, htqf_2, linestyle='-', color='red', label='htqf: u=0.6, d=1.2')
ax[2].axhline(0.0, linestyle='--', linewidth=0.5, color='k')
ax[2].axvline(0.0, linestyle='--', linewidth=0.5, color='k')
ax[2].set_xlim(-3, 3)
ax[2].set_ylim(-12, 12)
ax[2].set_xlabel('N(0,1)', fontsize=14)
ax[2].tick_params(axis='x', labelsize=12)
ax[2].tick_params(axis='y', labelsize=12)
ax[2].legend(fontsize=10)

fig.tight_layout()
plt.savefig('images_document/z_qq_plots.png')


# -------------------------------------------------------------------------------
# APPENDIX B: LOGISTIC SIGMOID FUNCTION
# -------------------------------------------------------------------------------

symbol = 'AAPL'
data_extracted = pd.read_csv('data/mode sl/datasets/' + symbol + '/data.csv')

log_return = np.diff(np.log(data_extracted['price']))

elle = 200
variances = []
standard_deviations = []

for pos in range(elle, len(log_return)):

    sample = log_return[pos - elle: pos]
    variances.append(np.var(sample))
    standard_deviations.append(np.std(sample))

plt.figure()
plt.plot(variances)

plt.figure()
plt.plot(standard_deviations)


# -------------------------------------------------------------------------------
# APPENDIX B: LOGISTIC SIGMOID FUNCTION
# -------------------------------------------------------------------------------


# Define the logistic sigmoid function

def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))


# Plot the logistic sigmoid function

z = np.arange(-10, 10, 0.01)
phi_z = sigmoid(z)

fig, ax = plt.subplots()
ax.plot(z, phi_z, color="blue")
ax.axhline(0.0, linestyle='--', linewidth=0.5, color='k')
ax.axhline(0.5, linestyle='--', linewidth=0.5, color='k')
ax.axhline(1.0, linestyle='--', linewidth=0.5, color='k')
ax.axvline(0.0, linestyle='--', linewidth=0.5, color='k')
ax.set_xlabel("Z", fontsize=14)
ax.set_ylabel(r"$\phi(Z)$", fontsize=14)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
fig.tight_layout()
plt.savefig('images_document/z_logistic_sigmoid_function.png')


# -------------------------------------------------------------------------------
# APPENDIX C: LOGISTIC COST FUNCTION
# -------------------------------------------------------------------------------


# Define the logistic cost function

def cost_1(z):

    return - np.log(sigmoid(z))


def cost_0(z):

    return - np.log(1 - sigmoid(z))


# Plot the logistic cost function

z = np.arange(-10, 10, 0.01)
phi_z = sigmoid(z)
c1 = cost_1(z)
c0 = cost_0(z)

fig, ax = plt.subplots()
ax.plot(phi_z, c1, color="blue", label="J(w) if y=1")
ax.plot(phi_z, c0, color="red", label="J(w) if y=0")
ax.set_xlabel(r"$\phi(Z)$", fontsize=14)
ax.set_ylabel("J(w)", fontsize=14)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.legend(loc="upper center", fontsize=12)
ax.set_xlim([0, 1])
ax.set_ylim([0, 5.1])
fig.tight_layout()
plt.savefig('images_document/z_logistic_cost_function.png')
