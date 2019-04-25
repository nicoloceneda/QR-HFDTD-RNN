""" data_loader
    -----------
    This script loads the trade data for a specific security and a time span.

"""


# IMPORT LIBRARIES AND/OR MODULES


import pandas as pd
import argparse
import taq_cleaner as tqc


# SQL query function to extract the data

def sql_query(table, symbol, nrows):                                                                                                        # TODO: Add option to select only certain cols

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


#

if __name__ == '__main__':

    # Parser

    parser = argparse.ArgumentParser(description='something')

    def check_nonnegative(value):
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError('Error: enter an integer greater or equal to zero')
        return ivalue

    parser.add_argument('--symbol', type=str, default='AAPL', help='Symbol to process.')
    parser.add_argument('--nrows', type=check_nonnegative, default=100, help='Number of rows to extract. If 0 then it loads all rows.')
    parser.add_argument('--max_attempts', type=int, default=3, help='Number of attempts to try to query the database.')
    parser.add_argument('--start_time', type=str, default='9:30', help='Start time to extract the data.')                                   # TODO: check the default values
    parser.add_argument('--end_time', type=str, default='15:30', help='End time to extract the data.')
    parser.add_argument('--start_date', type=str, default='2015-11-01', help='Start date to extract the data.')                             # TODO: check dates
    parser.add_argument('--end_date', type=str, default='2017-01-01', help='End date to extract the data.')

    args = parser.parse_args()

    #

    symbol = [args.symbol]





















    parser.add_argument('-save_output', action='store_true', help='Save the output file.')
    parser.add_argument('-output_name', type=str, default='trades', help='Name of output file.')
    parser.add_argument('-agg_unit', type=str, default='sec', help='Seconds or milliseconds on which to aggregate')
    parser.add_argument('-dt', type=int, default=100, help='Number of agg_units to aggregate')





    #

    try:
        import wrds
    except ImportError:
        print('An ImportError occurred. Run this script on the wrds cloud.')
    else:
        db = wrds.Connection()
        all_tables = db.list_tables(library='taqmsec')
        db.close()


    cleaning_params = TradeCleanParams(args)




    symbols = [args.symbol]

    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay

    us_businessday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    dates = pd.DatetimeIndex(start=args.start_date, end=args.end_date, freq=us_businessday)
    dates_str = [str(d)[0:10].replace('-','') for d in dates]



    for sym in symbols:

        for d_idx, date in enumerate(dates_str):

            if 'ctm_'+date in all_tables:
                print('symbol = {}, date = {}'.format(sym, date))
                query_trades = sql_query('taqmsec.'+date, sym, args.nrows)
                print('Loading data')
                trades, success_query_trades = query_loop(query_trades, args.max_attempts)

                if success_query_trades:
                    print('Cleaning data')

                    try:
                        trades_cleaner = tqc.taq_cleaner(trades, 'trades', cleaning_params)
                    except:
                        print('Something went wrong. Could not clean trades.')

                else:
                    print('Could not load data for {}. Skipping date and moving on'.format(date))

            else:
                print('Table not in databse.')