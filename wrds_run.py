import argparse
import pandas as pd


# Create a SQL query

def sql_query(table, symbol, nrows):

    if nrows > 0:
        query = 'SELECT * FROM {} WHERE sym_root = {} LIMIT {}'.format(table, symbol, nrows)
    else:
        query = 'SELECT * FROM {} WHERE sym_root = {}'.format(table, symbol)

    return query


# Parameters

class TradeCleanerParams:

    def __init__(self, args):
        self.first_hour = args.first_hour
        self.last_hour = args.last_hour
        self.col_types = {'EX': str, 'PRICE': float, 'SIZE': float}


class QuoteCleanerParams:

    def __init__(self, args):
        self.first_hour = args.first_hour
        self.last_hour = args.last_hour
        self.col_types = {'EX': str, 'BID': float, 'BIDSIZ': float, 'ASK': float, 'ASKSIZ': float}
        self.accepted_quote_cond = ['A', 'B', 'H', 'O', 'R', 'W']
        self.quote_null_filter = True
        self.min_spread = 0
        self.max_spread = 5


class NbboCleanerParams:

    def __init__(self, args):
        self.first_hour = args.first_hour
        self.last_hour = args.last_hour
        self.col_type = {'EX': float, 'BID': float, 'BIDSIZ': float, 'ASK': float, 'ASKSIZ': float}
        self.accepted_quote_cond = ['A', 'B', 'H', 'O', 'R', 'W']


class QuoteNbboParams:

    def __init__(self, args):
        self.pad_limit = args.pad_limit
        self.col_types = {'BBID': float, 'BBID_SIZE': float, 'TOTBID_SIZE': float, 'BASK': float, 'BASK_SIZE': float, 'TOTASK_SIZE': float,
                          'VWASK': float, 'VWBID': float}


class QuoteNbboMergeParams:

    def __init__(self, args):
        self.col_types = {'BID': float, 'BID_SIZ': float, 'ASK': float, 'ASK_SIZ': float}
        self.new_colnames = {'BID': 'BBID', 'BID_SIZ': 'BBID_SIZE', 'ASK': 'BASK', 'ASK_SIZ': 'BASK_SIZE'}


# Query loop function

def query_loop(query, max_trials):

    db = wrds_run.Connection()

    for trial in range(max_trials):
        try:
            data = db.raw_sql(query)
            db.close()
            return data, True
        except:
            print('Query failed')
            if trial < max_trials -1:
                print('Trying again')

    db.close()
    return None, False # TODO: Shouldn't this be inside the exception?


# Parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nbbo', action='store_true', help='Download the nbbo from the file. Otherwise it will be constructed.')            # 1
    parser.add_argument('-save_output', action='store_true', help='Save the output.')
    parser.add_argument('-output_name', type=str, default='merged_trade_quotes', help='Name of output file.')
    parser.add_argument('-save_only_aggregated', action='store_true', help='Save only aggregated data.')
    parser.add_argument('-nrows', type=int, default=10000, help='Number of rows to load. If 0 or negative load all rows.')
    parser.add_argument('-first_hour', type=str, default='9:30', help='First hour to accept trades/quotes.')
    parser.add_argument('-last_hour', type=str, default='15:30', help='last hour to accept trades/quotes.')
    parser.add_argument('-pad_limit', type=int, default=1000, help='Number rows to fill forward quotes if na')
    parser.add_argument('-max_trials', type=int, default=3, help='Number of times to try to query db')                                      # 9
    parser.add_argument('-agg_unit', type=str, default='sec', help='Seconds or milliseconds on which to aggregate')
    parser.add_argument('-dt', type=int, default=100, help='Number of agg_units to aggregate')
    parser.add_argument('-start_date', type=str, default='2015-11-01', help='First date to download')                                       # 12
    parser.add_argument('-end_date', type=str, default='2017-01-01', help='Last date to download')                                          # 13
    parser.add_argument('-symbol', type=str, default='AAPL', help='Symbol to process')                                                      # 14
    parser.add_argument('-debug', action='store_true', help='Debug flag')

    args = parser.parse_args()

    if args.debug:
        args.start_date = '2012-09-28'
        args.end_date = '2017-10-03'

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







