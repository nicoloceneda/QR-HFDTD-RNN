""" data_extraction.py
    ------------------
    This script extracts trade data for elected securities and a dates from the library 'taqmsec' contained in the wrds database.

    Contact: nicolo.ceneda@student.unisg.ch
    Last update: 26 April 2019

"""


# IMPORT LIBRARIES AND/OR MODULES


import pandas as pd
import argparse


# DATA EXTRACTION


# Function to run the SQL query and obviate occasional failures:

def query_sql(table, symbol, start_time, end_time):

    max_attempts = 2
    query = "SELECT * FROM {} WHERE sym_root = '{}' and time_m>= '{}' and time_m<= '{}'".format(table, symbol, start_time, end_time)        # TODO: Check how to handle multiple symbols

    dbi = wrds.Connection()

    for attempt in range(max_attempts):

        try:
            data = db.raw_sql(query)
        except Exception:
            if attempt < max_attempts - 1:
                print('The query failed: trying again.')
            else:
                print('The query failed and the max number of attempts has been reached.')
        else:
            dbi.close()
            return data, True

    dbi.close()
    return None, False


# Class to call all the parameters for a selected table (i.e. date) and symbol(s)

class TradesCleanedParameters:

    def __init__(self, arg):
        self.start_time = arg.start_time
        self.end_time = arg.end_time
        self.col_types = {'ex': str, 'price': float, 'size': float}                                                                         # TODO: Add the time column


# Class to clean the trades

class TradesCleaner:

    def __init__(self, data): # TODO: data_type <- 'trades' (string), cleaning_params (dictionary)

        self.data = data
        self.args = TradesCleanedParameters(args)

        if not set(self.args.col_types.keys()).issubset(self.data.columns): # TODO: if not {'ex', 'price', 'size'} is subset of
            raise ValueError('Either "ex", "price" or "size" is not a column in the queried data.')

        self.make_time_idx()
        self.data = self.data.between_time(self.args.start_time, self.args.end_time)

        self.clean_trades()

    def make_time_idx(self):

        datetime = self.data['date'].astype(str)+' '+self.data['time_m'].astype(str) # TODO: 2017-09-18 04:00:00.017518
        self.data = self.data.set_index(pd.DatetimeIndex(datetime)) # TODO: DatetimeIndex <--  dtype: object

    def clean_trades(self):

        self.data['tr_corr'][self.data['tr_corr']=='00'] = 0
        self.data = self.data[(self.data['tr_corr'] == 0) & (self.data['price'] > 0)]
        self.data = self.data[list(self.args.col_types.keys())]


# Command-line interface to extract data

parser = argparse.ArgumentParser(description='Command-line interface to extract the data')

parser.add_argument('-sy', '--symbols', type=str, default='AAPL', nargs='+', help='Symbol(s) to process.')
parser.add_argument('-sd', '--start_date', type=str, default='2003-09-10', help='Start date to extract the data.')                                 # TODO: check default dates
parser.add_argument('-ed', '--end_date', type=str, default='2018-04-20', help='End date to extract the data.')
parser.add_argument('-st', '--start_time', type=str, default='9:30', help='Start time to extract the data.')                                       # TODO: check the default values
parser.add_argument('-et', '--end_time', type=str, default='15:30', help='End time to extract the data.')
parser.add_argument('-bg', '--debug', action='store_true', help='Debuggin flag')

args = parser.parse_args()

# Debugging

if args.debug:
    args.start_date = '2018-09-17'
    args.end_date = '2018-09-17'
    args.start_time = '12:30:00'
    args.end_time = '12:31:00'

# List of all tables in 'taqmsec'

try:
    import wrds
except ImportError:
    raise ImportError('An Error occurred trying to run the script locally. Run this script on the wrds cloud.')
else:
    db = wrds.Connection()
    taqmsec_tables = db.list_tables(library='taqmsec')
    db.close()

# List of dates

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

us_businessday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
dates_idx = pd.DatetimeIndex(start=args.start_date, end=args.end_date, freq=us_businessday)
dates_list = [str(d)[0:10].replace('-', '') for d in dates_idx]

# List of symbols

symbols_list = args.symbols

# Extract data

for sym in symbols_list:

    for date in dates_list:

        if 'ctm_'+date in taqmsec_tables:

            print('Running a query with: symbol = {}, date = {}, nrows={}.'.format(sym, date, args.nrows))
            query_trades = query_sql('taqmsec.ctm_'+date, sym, args.nrows) # TODO: query_trades = "SELECT * FROM {taqmsec.ctm_ + 20170918} WHERE sym_root = '{}' LIMIT {} "
            queried_trades, success_query_trades = query_attempt(query_trades, args.max_attempts) # TODO: queried_trades = db.raw_sql(query_trades)

            if success_query_trades:
                print('Cleaning the data.')
                trades_cleaner = TradesCleaner()
            else:
                print('Could not load the data for security {} and date {}: skipping to the next date.'.format(sym, date))

        else:
            print('Could not find the table in database.')
