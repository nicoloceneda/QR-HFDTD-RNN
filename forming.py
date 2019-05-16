 # TODO: BananaSplit99?1357*7

import pandas as pd
import wrds
import pandas_market_calendars as mcal
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

db = wrds.Connection()


nasdaq = mcal.get_calendar('NASDAQ')
nasdaq_cal = nasdaq.schedule(start_date='2019-03-28', end_date='2019-03-29')
date_index = nasdaq_cal.index
date_list = [str(d)[:10].replace('-', '') for d in date_index]

symbol_list = ['AAPL', 'TSLA']

queried_trades = db.raw_sql("SELECT * FROM taqm_2019.ctm_20190328 WHERE sym_root = 'AAPL' and time_m >= '09:38' and time_m <= '09:48'")
output = queried_trades
queried_trades = db.raw_sql("SELECT * FROM taqm_2019.ctm_20190329 WHERE sym_root = 'AAPL' and time_m >= '09:38' and time_m <= '09:48'")
output = output.append(queried_trades)
queried_trades = db.raw_sql("SELECT * FROM taqm_2019.ctm_20190328 WHERE sym_root = 'TSLA' and time_m >= '09:38' and time_m <= '09:48'")
output = output.append(queried_trades)
queried_trades = db.raw_sql("SELECT * FROM taqm_2019.ctm_20190329 WHERE sym_root = 'TSLA' and time_m >= '09:38' and time_m <= '09:48'")
output = output.append(queried_trades)

output_ = output
my_copy = output

# --------------------------------------------------------

def graph_output(output_, symbol_list_, date_index_, usage):

    date_grid, symbol_grid = np.meshgrid(date_index_, symbol_list_)
    date_symbol_array = np.array([date_grid.ravel(), symbol_grid.ravel()]).T

    for date, symbol in date_symbol_array:

        x = output_.loc[(output_.loc[:, 'sym_root'] == symbol) & (pd.to_datetime(output_.loc[:, 'date']) == pd.to_datetime(date)), 'time_m']
        y = output_.loc[(output_.loc[:, 'sym_root'] == symbol) & (pd.to_datetime(output_.loc[:, 'date']) == pd.to_datetime(date)), 'price']

        xy = pd.DataFrame({'price': y})
        xy = xy.set_index(x)

        plt.figure()
        plt.plot(xy, color='b', linewidth=0.1)
        plt.title('{} {} {}'.format(symbol, str(pd.to_datetime(date))[:10], usage))
        plt.savefig('z_{}_{}.png'.format(symbol, str(pd.to_datetime(date))[:10], usage))

graph_output(output_, symbol_list, date_index, 'unfiltered')

def clean_heuristic_trades(output_):

    delta = 0.1
    k_list = np.arange(41, 121, 20, dtype='int64')
    y_list = np.arange(0.02, 0.08, 0.02)
    k_grid, y_grid = np.meshgrid(k_list, y_list)
    ky_array = np.array([k_grid.ravel(), y_grid.ravel()]).T

    outlier_frame = pd.DataFrame(columns=['out_num_max', 'k_max', 'y_max', 'out_num_min', 'k_min', 'y_min', 'out_num_score', 'k_score', 'y_score'], index=symbol_list)
    not_outlier_series_min = pd.Series(index=output_.index)
    not_outlier_series_max = pd.Series(index=output_.index)

    for symbol in symbol_list:

        count_1 = 0

        for k, y in ky_array: # TODO: use enumerate instead of count

            count_1 += 1
            outlier_num_sym = 0
            not_outlier_sym = pd.Series([])

            for date in date_list:

                price_sym_day = output_.loc[(output_.loc[:, 'sym_root'] == symbol) & (pd.to_datetime(output_.loc[:, 'date']) == pd.to_datetime(date)), 'price']
                price_sym_day_mean = pd.Series(index=list(range(price_sym_day.shape[0])))
                price_sym_day_std = pd.Series(index=list(range(price_sym_day.shape[0])))

                range_start = int((k - 1)/2)
                range_end = int(price_sym_day.shape[0] - (k - 1) / 2)

                for window_center in range(range_start, range_end):

                    window_start = window_center - range_start
                    window_end = window_center + range_start + 1
                    rolling_window = price_sym_day.iloc[window_start:window_end]
                    rolling_window_trimmed = rolling_window[(rolling_window > rolling_window.quantile(delta)) & (rolling_window < rolling_window.quantile(1-delta))]
                    price_sym_day_mean.iloc[window_center] = rolling_window_trimmed.mean()
                    price_sym_day_std.iloc[window_center] = rolling_window_trimmed.std()

                price_sym_day_mean.iloc[:range_start] = price_sym_day_mean.iloc[range_start]
                price_sym_day_mean.iloc[range_end:] = price_sym_day_mean.iloc[range_end - 1]
                price_sym_day_std.iloc[:range_start] = price_sym_day_std.iloc[range_start]
                price_sym_day_std.iloc[range_end:] = price_sym_day_std.iloc[range_end - 1]

                outlier_con_sym_day = (price_sym_day - price_sym_day_mean).abs() > 3 * price_sym_day_std + y
                outlier_num_sym_day = outlier_con_sym_day.sum()
                outlier_num_sym += outlier_num_sym_day

                not_outlier_con_sym_day = pd.Series((price_sym_day - price_sym_day_mean).abs() < 3 * price_sym_day_std + y)
                not_outlier_sym = not_outlier_sym.append(not_outlier_con_sym_day)

            if count_1 == 1:

                outlier_frame.loc[symbol, ['out_num_max', 'out_num_min']] = outlier_num_sym
                outlier_frame.loc[symbol, ['k_max', 'k_min']] = k
                outlier_frame.loc[symbol, ['y_max', 'y_min']] = y
                not_outlier_sym_min, not_outlier_sym_max = not_outlier_sym, not_outlier_sym

            elif outlier_num_sym < outlier_frame.loc[symbol, 'out_num_min']:

                outlier_frame.loc[symbol, 'out_num_min'] = outlier_num_sym
                outlier_frame.loc[symbol, 'k_min'] = k
                outlier_frame.loc[symbol, 'y_min'] = y
                not_outlier_sym_min = not_outlier_sym

            elif outlier_num_sym > outlier_frame.loc[symbol, 'out_num_max']:

                outlier_frame.loc[symbol, 'out_num_max']  = outlier_num_sym
                outlier_frame.loc[symbol, 'k_max'] = k
                outlier_frame.loc[symbol, 'y_max'] = y
                not_outlier_sym_max = not_outlier_sym

        not_outlier_series_min = not_outlier_series_min.append(not_outlier_sym_min)
        not_outlier_series_max = not_outlier_series_max.append(not_outlier_sym_max)

    output_filter_min = output_[not_outlier_series_min]
    output_filter_max = output_[not_outlier_series_max]

    return output_filter_min, output_filter_max


new_output_min, new_output_max = clean_heuristic_trades(output_)

graph_output(new_output_min, symbol_list, date_index, 'filtered_min')
graph_output(new_output_max, symbol_list, date_index, 'filtered_min')

plt.show()