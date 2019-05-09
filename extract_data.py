""" extract_data.py
    ---------------
    This script constructs the command line interface which is used to extracts trade data for selected dates, symbols and times from the
    wrds database.

    Contact: nicolo.ceneda@student.unisg.ch
    Last update: 03 May 2019

"""

# PROGRAM SETUP


# Import libraries and/or modules

import argparse
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import matplotlib.pyplot as plt
from colorama import *
init()


# Choose the setup

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Establish a connection to the wrds cloud

try:

    import wrds

except ImportError:

    raise ImportError('An error occurred trying to import the wrds library locally: run the script on the wrds cloud.')

else:

    db = wrds.Connection()


# COMMAND LINE INTERFACE AND INPUT CHECK


# Argument parser

min_start_date = '2003-09-10'
max_end_date = '2019-04-30'
min_start_time = '09:30'
max_end_time = '16:00'

parser = argparse.ArgumentParser(description='Command-line interface to extract the data')

parser.add_argument('-sl', '--symbol_list', metavar='', type=str, default=['AAPL'], nargs='+', help='List of symbols to extract.')
parser.add_argument('-sd', '--start_date', metavar='', type=str, default='{}'.format(min_start_date), help='Start date to extract the data.')
parser.add_argument('-ed', '--end_date', metavar='', type=str, default='{}'.format(max_end_date), help='End date to extract the data.')
parser.add_argument('-st', '--start_time', metavar='', type=str, default='{}'.format(min_start_time), help='Start time to extract the data.')
parser.add_argument('-et', '--end_time', metavar='', type=str, default='{}'.format(max_end_time), help='End time to extract the data.')     # TODO: check default times
parser.add_argument('-bg', '--debug', action='store_true', help='Flag to debug a single symbol; other params are set.')
parser.add_argument('-po', '--print_output', action='store_true', help='Flag to print the output.')
parser.add_argument('-go', '--graph_output', action='store_true', help='Flag to graph the output.')
parser.add_argument('-so', '--save_output', action='store_true', help='Flag to store the output.')
parser.add_argument('-on', '--name_output', metavar='', type=str, default='my_output', help='Name of the output file.')

args = parser.parse_args()


# Debug

if args.debug:

    args.symbol_list = ['AAPL', 'GOOG']
    args.start_date = '2019-03-19'
    args.end_date = '2019-03-21'
    args.start_time = '12:30:00'
    args.end_time = '12:30:10'
    args.print_output = True
    args.graph_output = True
    args.save_output = False

    print('You are debugging with: symbol_list: {}; start_date: {}; end_date: {}; start_time: {}; end_time: {}'.format(args.symbol_list,
          args.start_date, args.end_date, args.start_time, args.end_time))
    print()

else:

    print('You are querying with: symbol_list: {}; start_date: {}; end_date: {}; start_time: {}; end_time: {}'.format(args.symbol_list,
          args.start_date, args.end_date, args.start_time, args.end_time))


# Create list of symbols:

symbol_list = args.symbol_list


# Check dates and create list of dates:

print(Fore.RED)

if args.start_date > args.end_date:

    print('Error: Invalid start and end dates: chose a start date before the end date.')
    exit()

elif args.start_date < '{}'.format(min_start_date) and args.end_date < '{}'.format(max_end_date):

    print(Fore.RED + 'Error: Invalid start date: choose a date after {}.'.format(min_start_date))
    exit()

elif args.start_date > '{}'.format(min_start_date) and args.end_date > '{}'.format(max_end_date):

    print('Error: Invalid end date: choose a date before {}.'.format(max_end_date))
    exit()

elif args.start_date < '{}'.format(min_start_date) and args.end_date > '{}'.format(max_end_date):

    print('Error: Invalid start and end dates: choose dates between {} and {}.'.format(min_start_date, max_end_date))
    exit()

print(Fore.RESET)

us_businessday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
dates_idx = pd.date_range(start=args.start_date, end=args.end_date, freq=us_businessday)
date_list = [str(d)[:10].replace('-', '') for d in dates_idx]


# Check times:

print(Fore.RED)

if args.start_time > args.end_time:

    print('Error: Invalid start and end times: chose a start time before the end time.')
    exit()

elif args.start_time < '{}'.format(min_start_time) and args.end_time < '{}'.format(max_end_time):

    print('Error: Invalid start time: choose a time after {}.'.format(min_start_time))
    exit()

elif args.start_time > '{}'.format(min_start_time) and args.end_time > '{}'.format(max_end_time):

    print('Error: Invalid end time: choose a time before {}.'.format(max_end_time))
    exit()

elif args.start_time < '{}'.format(min_start_time) and args.end_time > '{}'.format(max_end_time):

    print('Error: Invalid start and end times: choose times between {} and {}.'.format(min_start_time, max_end_time))
    exit()

print(Fore.RESET)


# DATA EXTRACTION


# Function to run the SQL query and obviate occasional failures:

def query_sql(date_, symbol_, start_time_, end_time_):

    max_attempts = 2
    query = "SELECT * FROM taqm_{}.ctm_{} WHERE sym_root = '{}' and time_m >= '{}' and time_m <= '{}'".format(date_[:4], date_, symbol_,
            start_time_, end_time_)

    for attempt in range(max_attempts):

        try:

            queried_trades = db.raw_sql(query)

        except Exception:

            if attempt < max_attempts - 1:

                print(Fore.MAGENTA + 'Warning: The query failed: trying again.', 'orange' + Fore.RESET)

            else:

                print(Fore.MAGENTA + 'Warning: The query failed and the max number of attempts has been reached.', 'orange' + Fore.RESET)

        else:

            return queried_trades, True

    return None, False


# Extract the data

warning_queried_trades = []
warning_query_sql = []
warning_ctm_date = []
output = pd.DataFrame([])

for symbol in symbol_list:

    for date in date_list:

        all_tables = db.list_tables(library='taqm_{}'.format(date[:4]))

        if ('ctm_' + date) in all_tables:

            print('Running a query with: symbol: {}, date: {}, start_time: {}; end_time: {}.'. format(symbol, pd.to_datetime(date)
                  .strftime('%Y-%m-%d'), args.start_time, args.end_time))
            queried_trades, success_query_sql = query_sql(date, symbol, args.start_time, args.end_time)

            if success_query_sql:

                if queried_trades.shape[0] > 0:

                    print('Appending the queried trades to the output.')

                    if output.shape[0] == 0:

                       output = queried_trades

                    else:

                        output = output.append(queried_trades)

                else:

                    print(Fore.MAGENTA + 'Warning: Symbol {} did not trade on date {}: the warning has been recorded to "warning_queried_trades".'
                          .format(symbol, pd.to_datetime(date).strftime('%Y-%m-%d')) + Fore.RESET)
                    warning_queried_trades.append('{}+{}'.format(symbol, date))

            else:

                print(Fore.MAGENTA + 'Warning: The warning has been recorded to warning_query_sql".' + Fore.RESET)
                warning_query_sql.append('{}+{}'.format(symbol, date))


        else:

            print(Fore.MAGENTA + 'Warning: Could not find the table ctm_{} in the table list: the date has been removed from date_list and '
                  'the warning has been recorded to "warning_ctm_date".'.format(date) + Fore.RESET)
            date_list.remove(date)
            warning_ctm_date.append(date)


# Display outputs

print()
print(Fore.MAGENTA + 'warning_queried_trades' + Fore.RESET)
print(warning_queried_trades)

print()
print(Fore.MAGENTA + 'warning_query_sql' + Fore.RESET)
print(warning_query_sql)

print()
print(Fore.MAGENTA + 'warning_ctm_date' + Fore.RESET)
print(warning_ctm_date)

print()
print(output)


# DATA PLOTTING

if args.plot:

    for symbol in symbol_list:


        output['price'].plot()


# DATA CLEANING


# Clean data


#    def clean_trades(output_, k = 5):

#        length = [output_.shape[0]]

#        output_ = output_[output_['tr_corr'] == '00']
#        length.append(output_.shape[0])

#        output_ = output_[output_['tr_scond'] != 'Z']
#        length.append(output_.shape[0])

#        output_['outliers'] = np.absolute(output_['price'] - output_['price'].rolling(k, center=True).mean())

#        return output_, length


#    output_filtered, length = clean_trades(output)
#    print('The cleaning process shrunk the dataset as follows: original: {} -> after corr: {} -> after cond: {}'.format(length[0], length[1], length[2]))


# DATA PRINTING AND SAVING


# Print the output

if args.print_output:
    print(output_filtered)


# Save the output

if args.save_output:
    output_filtered.to_csv(args.name_output)


# PROGRAM SETUP


# Close the connection to the wrds cloud

db.close()

print('end')



plt.show()