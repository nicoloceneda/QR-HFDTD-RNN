

# -------------------------------------------------------------------------------
# 0. CODE SETUP
# -------------------------------------------------------------------------------


# Import the libraries

import numpy as np
import pandas as pd
import arch
import scipy.stats
import pyprind


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the extracted datasets

symbol = 'AAPL'
data_extracted = pd.read_csv('datasets/mode sl/datasets/' + symbol + '/data.csv')
data_extracted = data_extracted.iloc[:1000,:]

# Compute the returns

log_return = np.diff(np.log(data_extracted['price']))
data = pd.DataFrame({'log_return': log_return})
data.index = pd.DatetimeIndex(data_extracted['date'][1:].apply(str) + ' ' + data_extracted['time_m'][1:].apply(str))


def garch_model(elle_returns):

    model_garch = arch.arch_model(elle_returns, mean='Constant', dist='Normal', vol='Garch', p=1, q=1)
    fit_garch = model_garch.fit(disp='off')
    forecast_garch = fit_garch.forecast(horizon=1)
    var_garch = forecast_garch.variance.values[-1]
    vol_garch = np.sqrt(var_garch) / (10000)

    return vol_garch


elle = 200
Y = []

pbar = pyprind.ProgBar(50000)

for pos in range(elle, data.shape[0]):

    returns = 10000 * data.iloc[pos - elle: pos]
    volatility_garch = garch_model(returns)
    Y.append(volatility_garch)

    pbar.update()

Y2 = np.array(Y)

tau = np.concatenate(([0.01], np.divide(range(1, 20), 20), [0.99])).reshape(1, -1)
z_tau = scipy.stats.norm.ppf(tau, loc=0.0, scale=1.0)


res = Y2 * z_tau