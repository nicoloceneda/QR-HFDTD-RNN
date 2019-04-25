""" data_extraction.py
    ------------------
    This script extracts trade data for elected securities and a dates from the library 'taqmsec' contained in the wrds database.

    Contact: nicolo.ceneda@student.unisg.ch
    Last update: 26 April 2019

"""


# IMPORT LIBRARIES AND/OR MODULES


import pandas as pd
import argparse
import taq_cleaner as tqc


# DATA EXTRACTION


# SQL query function to extract the data

def sql_query(table, symbol, nrows):                                                                                                        # TODO: Add option to select only certain cols
                                                                                                                                            # TODO: Add the option to select the dates
    if nrows > 0:
        query = "SELECT * FROM {} WHERE sym_root = '{}' LIMIT {}".format(table, symbol, nrows)
    else:
        query = "SELECT * FROM {} WHERE sym_root = '{}'".format(table, symbol)

    return query


# Query attempt function to obviate occasional failures of the raw_sql(query) command

def query_attempt(query, max_attempts):                                                                                                     # TODO: WRDS must be imported before calling

    db = wrds.Connection()

    for attempt in range(max_attempts):

        try:
            data = db.raw_sql(query)
        except Exception:
            if attempt < max_attempts - 1:
                print('The query failed: trying again.')
            else:
                print('The query failed and the max number of attempts has been reached.')
        else:
            db.close()
            return data, True

    db.close()
    return None, False


# Parameters

class TradeCleanParams:
    def __init__(self, arg):
        self.start_time = arg.first_hour
        self.end_time = arg.last_hour
        self.col_types = {'ex': str, 'price': float, 'size': float}


# Command-line interface to extract data

if __name__ == '__main__':

    # Parser

    parser = argparse.ArgumentParser(description='something')

    def check_nonnegative(value):
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError('Error: enter an integer greater or equal to zero')
        return ivalue

    parser.add_argument('--symbols', type=str, default='AAPL', nargs='+', help='Symbol(s) to process.')
    parser.add_argument('--nrows', type=check_nonnegative, default=100, help='Number of rows to extract. If 0 then it loads all rows.')
    parser.add_argument('--max_attempts', type=int, default=3, help='Number of attempts to try to query the database.')
    parser.add_argument('--start_time', type=str, default='9:30', help='Start time to extract the data.')                                   # TODO: check the default values
    parser.add_argument('--end_time', type=str, default='15:30', help='End time to extract the data.')
    parser.add_argument('--start_date', type=str, default='2015-11-01', help='Start date to extract the data.')                             # TODO: check dates
    parser.add_argument('--end_date', type=str, default='2017-01-01', help='End date to extract the data.')

    args = parser.parse_args()

    # Cleaning parameters

    cleaning_params = {'trades': TradeCleanParams(args)}                                                                                      # TODO: CHECK

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
                query_trades = sql_query('taqmsec.ctm_'+date, sym, args.nrows)
                queried_trades, success_query_trades = query_attempt(query_trades, args.max_attempts)

                if success_query_trades:
                    print('Cleaning the data.')
                    trades_cleaner = tqc.taq_cleaner(queried_trades, 'trades', cleaning_params)
                else:
                    print('Could not load the data for security {} and date {}: skipping to the next date.'.format(sym, date))

            else:
                print('Could not find the table in database.')
