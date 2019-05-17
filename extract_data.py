""" extract_data.py
    ---------------
    This script constructs the command line interface which is used to extract and clean trade data for selected symbols, dates and times
    from the wrds database.

    Contact: nicolo.ceneda@student.unisg.ch
    Last update: 17 May 2019

"""


# ------------------------------------------------------------------------------------------------------------------------------------------
# PROGRAM SETUP
# ------------------------------------------------------------------------------------------------------------------------------------------


# Import the libraries and the modules

import argparse
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt


# Import the functions from the functions file

from extract_data_functions import section, graph_output, reset_index


# Set the displayed size of pandas objects

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Establish a connection to the wrds cloud

try:

    import wrds

except ImportError:

    raise ImportError('An error occurred trying to import the wrds library locally: run the script on the wrds cloud.')

else:

    db = wrds.Connection()


# ------------------------------------------------------------------------------------------------------------------------------------------
# COMMAND LINE INTERFACE AND INPUT CHECK
# ------------------------------------------------------------------------------------------------------------------------------------------


# Define the commands available in the command line interface

min_start_date = '2003-09-10'
max_end_date = '2019-05-10'
min_start_time = '09:30:00'
max_end_time = '16:00:00'

parser = argparse.ArgumentParser(description='Command-line interface to extract trade data')

parser.add_argument('-sl', '--symbol_list', metavar='', type=str, default=['AAPL'], nargs='+', help='List of symbols to extract.')
parser.add_argument('-sd', '--start_date', metavar='', type=str, default='{}'.format(min_start_date), help='Start date to extract the data.')
parser.add_argument('-ed', '--end_date', metavar='', type=str, default='{}'.format(max_end_date), help='End date to extract the data.')
parser.add_argument('-st', '--start_time', metavar='', type=str, default='{}'.format(min_start_time), help='Start time to extract the data.')
parser.add_argument('-et', '--end_time', metavar='', type=str, default='{}'.format(max_end_time), help='End time to extract the data.')
parser.add_argument('-bg', '--debug', action='store_true', help='Flag to debug the program.')
parser.add_argument('-po', '--print_output', action='store_true', help='Flag to print the output.')
parser.add_argument('-go', '--graph_output', action='store_true', help='Flag to graph the output.')
parser.add_argument('-so', '--save_output', action='store_true', help='Flag to store the output.')
parser.add_argument('-on', '--name_output', metavar='', type=str, default='my_output', help='Name of the output file.')

args = parser.parse_args()


# Define the debug settings

if args.debug:

    args.symbol_list = ['AAPL', 'TSLA']
    args.start_date = '2019-03-28'
    args.end_date = '2019-03-29'
    args.start_time = '09:38:00'
    args.end_time = '09:48:00'
    args.print_output = True
    args.graph_output = True
    args.save_output = False

    section('You are debugging with: symbol_list: {}; start_date: {}; end_date: {}; start_time: {}; end_time: {}'.format(args.symbol_list,
          args.start_date, args.end_date, args.start_time, args.end_time))

else:

    section('You are querying with: symbol_list: {}; start_date: {}; end_date: {}; start_time: {}; end_time: {}'.format(args.symbol_list,
          args.start_date, args.end_date, args.start_time, args.end_time))


# Create the list of the input symbols:

symbol_list = args.symbol_list


# Check the validity of the input dates and create the list of dates:

if args.start_date > args.end_date:

    print('*** ERROR: Invalid start and end dates: chose a start date before the end date.')
    exit()

elif args.start_date < '{}'.format(min_start_date) and args.end_date < '{}'.format(max_end_date):

    print('*** ERROR: Invalid start date: choose a date after {}.'.format(min_start_date))
    exit()

elif args.start_date > '{}'.format(min_start_date) and args.end_date > '{}'.format(max_end_date):

    print('*** ERROR: Invalid end date: choose a date before {}.'.format(max_end_date))
    exit()

elif args.start_date < '{}'.format(min_start_date) and args.end_date > '{}'.format(max_end_date):

    print('*** ERROR: Invalid start and end dates: choose dates between {} and {}.'.format(min_start_date, max_end_date))
    exit()

nasdaq = mcal.get_calendar('NASDAQ')
nasdaq_cal = nasdaq.schedule(start_date=args.start_date, end_date=args.end_date)
date_index = nasdaq_cal.index
date_list = [str(d)[:10].replace('-', '') for d in date_index]


# Check the validity of the input times:

if args.start_time > args.end_time:

    print('*** ERROR: Invalid start and end times: chose a start time before the end time.')
    exit()

elif args.start_time < '{}'.format(min_start_time) and args.end_time < '{}'.format(max_end_time):

    print('*** ERROR: Invalid start time: choose a time after {}.'.format(min_start_time))
    exit()

elif args.start_time > '{}'.format(min_start_time) and args.end_time > '{}'.format(max_end_time):

    print('*** ERROR: Invalid end time: choose a time before {}.'.format(max_end_time))
    exit()

elif args.start_time < '{}'.format(min_start_time) and args.end_time > '{}'.format(max_end_time):

    print('*** ERROR: Invalid start and end times: choose times between {} and {}.'.format(min_start_time, max_end_time))
    exit()


# ------------------------------------------------------------------------------------------------------------------------------------------
# DATA EXTRACTION
# ------------------------------------------------------------------------------------------------------------------------------------------


# Create a function to run the SQL query and obviate occasional failures:

def query_sql(date_, symbol_, start_time_, end_time_):

    max_attempts = 2
    query = "SELECT * FROM taqm_{}.ctm_{} WHERE sym_root = '{}' and time_m >= '{}' and time_m <= '{}'".format(date_[:4], date_, symbol_,
            start_time_, end_time_)

    for attempt in range(max_attempts):

        try:

            queried_trades = db.raw_sql(query)

        except Exception:

            if attempt < max_attempts - 1:

                print('*** WARNING: The query failed: trying again.')

            else:

                print('*** WARNING: The query failed and the max number of attempts has been reached.')

        else:

            return queried_trades, True

    return None, False


# Create a function to check the min and max number of observations for each symbol

def n_obs(queried_trades_, date_):

    global count_2, min_n_obs, min_n_obs_day, max_n_obs, max_n_obs_day
    count_2 += 1
    obs = queried_trades_.shape[0]

    if count_2 == 1:

        min_n_obs = obs
        n_obs_table.loc[count_1, 'min_n_obs'] = min_n_obs
        n_obs_table.loc[count_1, 'min_n_obs_day'] = pd.to_datetime(date_).strftime('%Y-%m-%d')

        max_n_obs = obs
        n_obs_table.loc[count_1, 'max_n_obs'] = max_n_obs
        n_obs_table.loc[count_1, 'max_n_obs_day'] = pd.to_datetime(date_).strftime('%Y-%m-%d')

    elif obs < min_n_obs:

        min_n_obs = obs
        n_obs_table.loc[count_1, 'min_n_obs'] = min_n_obs
        n_obs_table.loc[count_1, 'min_n_obs_day'] = pd.to_datetime(date_).strftime('%Y-%m-%d')

    elif obs > max_n_obs:

        max_n_obs = obs
        n_obs_table.loc[count_1, 'max_n_obs'] = max_n_obs
        n_obs_table.loc[count_1, 'max_n_obs_day'] = pd.to_datetime(date_).strftime('%Y-%m-%d')


# Run the SQL queries and compute the min and max number of observations for each queried symbol

warning_queried_trades = []
warning_query_sql = []
warning_ctm_date = []

n_obs_table = pd.DataFrame({'symbol': [], 'min_n_obs': [], 'min_n_obs_day': [], 'max_n_obs': [], 'max_n_obs_day': []})

output = pd.DataFrame([])

remove_dates = []

for count_1, symbol in enumerate(symbol_list):

    min_n_obs = None
    min_n_obs_day = None
    max_n_obs = None
    max_n_obs_day = None
    n_obs_table.loc[count_1, 'symbol'] = symbol
    count_2 = 0

    for date in date_list:    

        print('Running a query with: symbol: {}, date: {}, start_time: {}; end_time: {}.'.format(symbol, pd.to_datetime(date)
              .strftime('%Y-%m-%d'), args.start_time, args.end_time))

        all_tables = db.list_tables(library='taqm_{}'.format(date[:4]))

        if ('ctm_' + date) in all_tables:

            queried_trades, success_query_sql = query_sql(date, symbol, args.start_time, args.end_time)

            if success_query_sql:

                if queried_trades.shape[0] > 0:

                    print('Appending the queried trades to the output.')

                    if output.shape[0] > 0:

                        output = output.append(queried_trades)

                    else:

                        output = queried_trades

                    n_obs(queried_trades, date)

                else:

                    print('*** WARNING: Symbol {} did not trade on date {}: the warning has been recorded to "warning_queried_trades".'
                          .format(symbol, pd.to_datetime(date).strftime('%Y-%m-%d')))
                    warning_queried_trades.append('{}+{}'.format(symbol, date))

            else:

                print('*** WARNING: The warning has been recorded to warning_query_sql".')
                warning_query_sql.append('{}+{}'.format(symbol, date))

        else:

            print('*** WARNING: Could not find the table ctm_{} in the table list: the date has been removed from date_list; '
                  'the warning has been recorded to "warning_ctm_date".'.format(date))
            remove_dates.append(date)
            warning_ctm_date.append(date)

    date_list = [d for d in date_list if d not in remove_dates]


# ------------------------------------------------------------------------------------------------------------------------------------------
# DISPLAY THE RESULTS OF DATA EXTRACTION
# ------------------------------------------------------------------------------------------------------------------------------------------


# Display the log of the warnings

section('Log of the raised warnings')

print('*** LOG: warning_queried_trades:')
print(warning_queried_trades)

print('*** LOG: warning_ctm_date:')
print(warning_ctm_date)

print('*** LOG: warning_query_sql:')
print(warning_query_sql)


# Display the dataframe with the min and max number of observations for each symbol

section('Min and max number of observations for each queried symbol')

print(n_obs_table)


# Display the dataframe of the queried trades

section('Queried data')

if args.print_output:

    print(output)

else:

    print('"Print output" is not active')


# Display the plots of the queried trades

if args.graph_output:

    graph_output(output, symbol_list, date_index, 'unfiltered')


# ------------------------------------------------------------------------------------------------------------------------------------------
# DATA CLEANING
# ------------------------------------------------------------------------------------------------------------------------------------------


# Create a function to check the queried trades for 'tr_corr' and 'tr_scond'

def clean_time_trades(output_):

    tr_corr_check = output_['tr_corr'] == '00'

    tr_scond_check = pd.Series(index=output_.index)
    char_allowed = {'@', 'A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'M', 'N', 'O', 'Q', 'R', 'S', 'V', 'W', 'Y', '1', '4', '5', '6', '7', '8', '9'}
    char_forbidden = {'G', 'L', 'P', 'T', 'U', 'X', 'Z'}
    char_recognized = char_allowed | char_forbidden
    char_unseen = []

    for row in range(output_.shape[0]):

        element = output_.iloc[row, 5].replace(" ", "")

        if any((char in char_forbidden) for char in element) & all((char in char_recognized) for char in element):

            tr_scond_check.iloc[row] = False

        elif all((char in char_allowed) for char in element):

            tr_scond_check.iloc[row] = True

        elif any((char not in char_recognized) for char in element):

            tr_scond_check.iloc[row] = False
            char_unseen.append(row)

    if len(char_unseen) > 0:

        print('*** LOG: rows with unseen conditions:')
        print(char_unseen)

    return tr_corr_check, tr_scond_check


# Create a function to check the queried trades for outliers

def clean_heuristic_trades(output_, symbol_list_, date_list_):

    delta = 0.1
    k_list = np.arange(41, 121, 20, dtype='int64')
    y_list = np.arange(0.02, 0.08, 0.02)
    k_grid, y_grid = np.meshgrid(k_list, y_list)
    ky_array = np.array([k_grid.ravel(), y_grid.ravel()]).T

    outlier_frame = pd.DataFrame(columns=['out_num_max', 'k_max', 'y_max', 'score_max', 'out_num_min', 'k_min', 'y_min', 'score_min'], index=symbol_list_)
    not_outlier_series = pd.Series([])

    for symbol in symbol_list_:

        count_1 = 0

        for k, y in ky_array:

            count_1 += 1
            outlier_num_sym = 0
            outlier_val_sym = 0
            not_outlier_sym = pd.Series([])

            for date in date_list_:

                price_sym_day = output_.loc[(output_['sym_root'] == symbol) & (pd.to_datetime(output_['date']) == pd.to_datetime(date)), 'price'] # 30 40
                price_sym_day_mean = pd.Series(index=price_sym_day.index)
                price_sym_day_std = pd.Series(index=price_sym_day.index)

                range_start = int((k - 1)/2) # 20
                range_end = int(price_sym_day.shape[0] - (k - 1) / 2) #

                for window_center in range(range_start, range_end):

                    window_start = window_center - range_start # 0 1 2 3
                    window_end = window_center + range_start + 1 # 41 42 43
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

                outlier_val_sym_day = (price_sym_day - price_sym_day_mean).abs()
                outlier_val_sym += (outlier_val_sym_day[outlier_con_sym_day]).sum()

                not_outlier_con_sym_day = pd.Series((price_sym_day - price_sym_day_mean).abs() < 3 * price_sym_day_std + y)
                not_outlier_sym = not_outlier_sym.append(not_outlier_con_sym_day)

            if count_1 == 1:

                outlier_frame.loc[symbol, ['out_num_max', 'out_num_min']] = outlier_num_sym
                outlier_frame.loc[symbol, ['score_max', 'score_min']] = outlier_val_sym / outlier_num_sym
                outlier_frame.loc[symbol, ['k_max', 'k_min']] = k
                outlier_frame.loc[symbol, ['y_max', 'y_min']] = y
                not_outlier_sym_min, not_outlier_sym_max = not_outlier_sym, not_outlier_sym

            elif outlier_num_sym < outlier_frame.loc[symbol, 'out_num_min']:

                outlier_frame.loc[symbol, 'out_num_min'] = outlier_num_sym
                outlier_frame.loc[symbol, 'score_min'] = outlier_val_sym / outlier_num_sym
                outlier_frame.loc[symbol, 'k_min'] = k
                outlier_frame.loc[symbol, 'y_min'] = y
                not_outlier_sym_min = not_outlier_sym

            elif outlier_num_sym > outlier_frame.loc[symbol, 'out_num_max']:

                outlier_frame.loc[symbol, 'out_num_max'] = outlier_num_sym
                outlier_frame.loc[symbol, 'score_max'] = outlier_val_sym / outlier_num_sym
                outlier_frame.loc[symbol, 'k_max'] = k
                outlier_frame.loc[symbol, 'y_max'] = y
                not_outlier_sym_max = not_outlier_sym

        if outlier_frame.loc[symbol, 'score_min'] > outlier_frame.loc[symbol, 'score_max']:

            not_outlier_series = not_outlier_series.append(not_outlier_sym_min)

        else:

            not_outlier_series = not_outlier_series.append(not_outlier_sym_max)

    return not_outlier_series


# ------------------------------------------------------------------------------------------------------------------------------------------
# DISPLAY THE RESULTS OF DATA CLEANING
# ------------------------------------------------------------------------------------------------------------------------------------------


# Compute the filter to clean the data from unwanted 'tr_corr' and 'tr_scond'

filter_tr_corr, filter_tr_scond = clean_time_trades(output)


# Compute the filter to clean the data from outliers

filter_outlier = clean_heuristic_trades(output, symbol_list, date_list)


# Clean the data

output_filtered = output[filter_tr_corr & filter_tr_scond & filter_outlier]


# Display the cleaned dataframe of the queried trades

section('Cleaned data')

if args.print_output:

    print(output_filtered)

else:

    print('"Print output" is not active')


# Display the plots of the cleaned trades

if args.graph_output:

    graph_output(output_filtered, symbol_list, date_index, 'filtered')




# PROGRAM SETUP


# Close the connection to the wrds cloud

db.close()

print()
print('End of Execution')

plt.show()


# TODO: Addition of dividend and/or split adjustments

# TODO: Filter the data as outlined in the CF File Description Section 3.2

# TODO: Check for different classes of shares (ex. GOOG and GOOG L)

# TODO: Calculation of statistics such as the opening and closing prices from the primary market, the high and low from the consolidated
#  market, share volume from the primary market, and consolidated share volume.

# TODO: Plot the charts.

# TODO: Solve the suffix for GOOG