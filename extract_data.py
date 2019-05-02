# IMPORT LIBRARIES AND/OR MODULES


import argparse
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


# WRDS CLOUD


# Establish a connection to the wrds cloud

try:
    import wrds

except ImportError:
    raise ImportError('An error occurred trying to import the wrds library locally: run the script on the wrds cloud.')

else:
    db = wrds.Connection()


# COMMAND LINE INTERFACE


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
parser.add_argument('-et', '--end_time', metavar='', type=str, default='{}'.format(max_end_time), help='End time to extract the data.')                       # TODO: check default times
parser.add_argument('-bg', '--debug', action='store_true', help='Flag to debug a single symbol; other params are set.')
parser.add_argument('-so', '--save_output', action='store_true', help='Flag to store the output.')
parser.add_argument('-on', '--output_name', metavar='', type=str, help='Name of the output file.')

args = parser.parse_args()


# Create list of symbols:

symbol_list = args.symbol_list


# Check dates and create list of dates:

if args.start_date > args.end_date:
    print(KeyError('Invalid start and end dates: chose a start date before the end date.'))

elif args.start_date < '{}'.format(min_start_date) and args.end_date < '{}'.format(max_end_date):
    print(KeyError('Invalid start date: choose a date after {}.'.format(min_start_date)))

elif args.start_date > '{}'.format(min_start_date) and args.end_date > '{}'.format(max_end_date):
    print(KeyError('Invalid end date: choose a date before {}.'.format(max_end_date)))

elif args.start_date < '{}'.format(min_start_date) and args.end_date > '{}'.format(max_end_date):
    print(KeyError('Invalid start and end dates: choose dates between {} and {}.'.format(min_start_date, max_end_date)))

us_businessday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
dates_idx = pd.date_range(start=args.start_date, end=args.end_date, freq=us_businessday)
date_list = [str(d)[:10].replace('-', '') for d in dates_idx]


# Check times:

if args.start_time > args.end_time:
    print(KeyError('Invalid start and end times: chose a start time before the end time.'))

elif args.start_time < '{}'.format(min_start_time) and args.end_time < '{}'.format(max_end_time):
    print(KeyError('Invalid start time: choose a time after {}.'.format(min_start_time)))

elif args.start_time > '{}'.format(min_start_time) and args.end_time > '{}'.format(max_end_time):
    print(KeyError('Invalid end time: choose a time before {}.'.format(max_end_time)))

elif args.start_time < '{}'.format(min_start_time) and args.end_time > '{}'.format(max_end_time):
    print(KeyError('Invalid start and end times: choose times between {} and {}.'.format(min_start_time, max_end_time)))


# Debug

if args.debug:

    if len(args.symbol_list) > 1:
        raise KeyError('You are debugging: insert one symbol at the time.')

    else:
        args.start_date = '2019-04-30'
        args.end_date = '2019-04-30'
        args.start_time = '12:30'
        args.end_time = '13:00'

        print('You are debugging with: symbol_list: {}; start_date: {}; end_date: {}; start_time: {}; end_time: {}'.format(args.symbol_list,
              args.start_date, args.end_date, args.start_time, args.end_time))

else:
    print('You are querying with: symbol_list: {}; start_date: {}; end_date: {}; start_time: {}; end_time: {}'.format(args.symbol_list,
          args.start_date, args.end_date, args.start_time, args.end_time))
    print()


# DATA EXTRACTION


# Function to run the SQL query and obviate occasional failures:

def query_sql(date_, symbol_, start_time_, end_time_):

    max_attempts = 2
    query = "SELECT * FROM taqm_{}.ctm_{} WHERE sym_root = '{}' and time_m >= '{}' and time_m <= '{}'".format(date_[:4], date_, symbol_,
            start_time_, end_time_)

    for attempt in range(max_attempts):

        try:
            queried_trades_ = db.raw_sql(query)

        except Exception:

            if attempt < max_attempts - 1:
                print('The query failed: trying again.')
            else:
                print('The query failed and the max number of attempts has been reached.')

        else:
            return queried_trades_, True

    return None, False

# Extract the data

all_libraries = db.list_libraries()
output = pd.DataFrame([])

for date in date_list:

    all_tables = db.list_tables(library='taqm_{}'.format(date[:4]))

    for symbol in symbol_list:

        if ('ctm_' + date) in all_tables:
            print('Running a query with: date: {}, symbol: {}, start_time: {}; end_time: {}'. format(date, symbol, args.start_time, args.end_time))
            queried_trades, success_query_sql = query_sql(date, symbol, args.start_time, args.end_time)
            print(queried_trades)

        else:
            print('Could not run a query with: date: {}, symbol: {}'. format(date, symbol))


# WRDS CLOUD


# Close the connection to the wrds cloud

db.close()

print('end')