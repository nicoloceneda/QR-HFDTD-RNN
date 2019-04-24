import argparse
import pandas as pd
import wrds


# SQL query

def sql_query( table, symbol, nrows):                                                                                                       # TODO: Add option to select only certain cols

    if nrows > 0:
        query = 'SELECT * FROM {} WHERE sym_root = {} LIMIT'.format(table, symbol, nrows)
    else:
        query = 'SELECT * FROM {} WHERE sym_root = {}'.format(table, symbol)

    return query


# Query attempt function

def query_loop(query, max_attempts):

    db = wrds.Connection()

    for attempt in range(max_attempts):

        try:
            data = db.raw_sql(query)
        except OperationalError:
            if attempt < max_attempts-1:
                print('OperationalError. Trying again')
            else:
                print('OperationalError. Max attempts reached')
        except Exception:
            if attempt < max_attempts - 1:
                print('Other error. Trying again')
            else:
                print('Other error. Max attempts reached')
        else:
            return data, True
        finally:
            db.close()

    return None, False


# Parameters

class TradeCleanParams:

    def __init__(self, args):
        self.first_hour = args.first_hour
        self.last_hour = args.last_hour
        self.col_types = {'ex': str, 'price': float, 'size': float}


# Parser

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='something')

    def check_positive(value):
        if value < 0:
            raise argparse.ArgumentTypeError('Error: enter an integer >= 0')
        return value

    parser.add_argument('-symbol', type=str, default='AAPL', help='Symbol to process')
    parser.add_argument('-nrows', type=check_positive, default=100, help='Number of rows to load. If 0 then it loads all rows.')

    parser.add_argument('-start_time', type=str, default='9:30', help='Start time to load trades')                                          # TODO: check the hours
    parser.add_argument('-end_time', type=str, default='15:30', help='End hour to load trades')
    parser.add_argument('-start_date', type=str, default='2015-11-01', help='Start date to load trades')                                    # TODO: check dates
    parser.add_argument('-end_date', type=str, default='2017-01-01', help='End date to load trades')

    parser.add_argument('-max_attempts', type=int, default=3, help='Number of attempts to try to query the database.')

    parser.add_argument('-save_output', action='store_true', help='Save the output file.')
    parser.add_argument('-output_name', type=str, default='trades', help='Name of output file.')
    parser.add_argument('-agg_unit', type=str, default='sec', help='Seconds or milliseconds on which to aggregate')
    parser.add_argument('-dt', type=int, default=100, help='Number of agg_units to aggregate')

    args = parser.parse_args()

    symbols = [args.symbol]                                                                                                                 # TODO: remove these three lines?
    print(args.start_date)
    print(args.end_date)

    symbols = [args.symbol]

    try:
        import wrds_run
    except ImportError:
        print('This script has to be run on the wrds cloud')

    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    us_businessday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    dates = pd.DatetimeIndex(start=args.start_date, end=args.end_date, freq=us_businessday)
    dates_str = [str(d)[0:10].replace('-','') for d in dates]

    db.Connction()
    all_tables = db.list_tables(library='taqmsec')
    db.close()

    cleaning_params = {'trades': TradeCleanerParams, 'quotes': QuoteCleanerParams(args)}
    nbbo_params = {'quotes': QuoteNbboParams}

    for sym in symbols:
        dfs = []
        month = dates[0].month

        for d_idx, date in enumerate(dates_str):

            if not args.nbbo:
                table_quotes = 'cqm_' + date
                table_trades = 'ctm_' + date

                if table_quotes in all_tables and table_trades in all_tables:
                    print('symbol = {}, date = {}'.format(sym, date))
                    query_quotes = sql_query('taqmsec.'+table_quotes, sym, args.nrows)
                    query_trades = sql_query('taqmsec.'+table_trades, sym, args.nrows)
                    print('Loading data')
                    quotes, success_query_quotes = query_loop(query_quotes, args.max_trials)
                    trades, success_query_trades = query_loop(query_trades, args.max_trials)

                    if success_query_quotes and success_query_trades:
                        print('Cleaning, merging')

                        try:
                            quotes_cleaner = tqc.taq_cleaner(quotes, 'quotes', cleaning_params)  # TODO: understand TAQ cleaner
                            quotes = quotes_cleaner.data
                            nbbo_maker = tqnm.taq_nbbo_maker({'quotes': quotes}, nbbo_params, mode='quotes')  # TODO: understand nbbo maker
                            nbbo = nbbo_maker.data
                            nbbo['SYMBOL'] = sym
                            trades_cleaner = tqc.taq_cleaner(trades, 'trades', cleaning_params)
                            nbbo = pd.merge(nbbo, trades_cleaner.data, left_index=True, right_index=True, how='outer')
                            dfs.append(nbbo)

                        except:
                            print('Something went wrong. Could not process quotes and trades')

                    else:
                        print('Could not load the data for {}. Skipping date and moving on.'.format(date))

                else:
                    print('Table not in database.')

            else:
                table_nbbo = 'nbbo_'+date

                if table_nbbo in all_tables:
                    query_nbbo = form_sql_query(db_name + '.' + table_quotes, sym, args.nrows)
                    nbbo = db.raw_sql(query_nbbo)
                    nbbo.columns = [c.upper() for c in nbbo.columns]
                else:
                    print('Table not in database.')

            save_and_agg = True

            if save_and_agg:

                if len(dfs) > 0:
                    print('Save and aggregate: sym = {}, month = {}'.format(sym, month)
                    df = pd.concat(dfs)
                    fname = args.save_fname + '_' + sym + '_' + str(dates[d_idx].year) + '_' + str(dates[d_idx].month) + '_' + str(dates[d_idx].day)
                    path = os.path.join(os.getcwd(), 'output')
                    if not args.save_only_agg:
                        df.to_pickle(
                            os.path.join(path, fname + '.gzip'))

                    params_agg = {'agg_unit' : args.agg_unit,'dt' : args.dt}
                    dag = aggregator.data_aggregator(path, fname, params_agg, data=df)
                    dag.aggregate()
                    dag.save_agg()

                    dfs = []
                    if d_idx < len(dates) - 1:
                        month = dates[d_idx+1].month
                    else:
                        print('REACHED LAST DATE. DONE.')







