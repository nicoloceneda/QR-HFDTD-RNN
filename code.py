""" PARSIMONIOUS QUANTILE REGRESSION OF FINANCIAL ASSET TAIL DYNAMICS VIA SEQUENTIAL LEARNING

    Author:       Nicolo Ceneda
    Contact:      nicolo.ceneda@student.unisg.ch
    Institution:  University of St Gallen
    Course:       Master of Banking and Finance


    README:

    1) Before executing this program, run the following code in the console to create a pgpass file:

    import wrds
    db = wrds.Connection(wrds_username='ceneda')
    db.create_pgpass_file()

    2) To start an interactive section on the WRDS cloud, run the following commands in the terminal:

    ssh ceneda@wrds-cloud.wharton.upenn.edu
    qrsh
    ipython3
    import wrds
    db = wrds.Connection()

"""


# IMPORT LIBRARIES/MODULES


import wrds
import pandas as pd

# Library settings

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)


# DATA


# Establish a connection to wrds

db = wrds.Connection(wrds_username="ceneda")

# Data structure

sorted(db.list_libraries())

sorted(db.list_tables(library="taqmsec"))

db.describe_table(library="taqmsec", table="cqm_20180102")
db.describe_table(library="taqmsec", table="ctm_20180102")
db.describe_table(library="taqmsec", table="nbbom_20180102")
db.describe_table(library="taqmsec", table="mastm_20180102")

# Data quering data = db.raw_sql('SELECT comnam,cusip,namedt,nameenddt FROM crsp.stocknames LIMIT 2')

trades = db.raw_sql("SELECT time_m, size, price FROM taqmsec.ctm_20180102 LIMIT 10", date_cols=['date'])
print(trades.tail())

data = db.raw_sql('select date, time_m, bid, bidsiz, ask, asksiz, best_bid, best_bidsiz, best_ask, best_asksiz, qu_seqnum from taqmsec.nbbom_20180102 LIMIT 100;', date_cols=['date'])
data

mas = db.raw_sql('select date, symbol_15, cusip from taqmsec.mastm_20180102 LIMIT 100;', date_cols=['date'])
mas

parm = {'tickers': 'AAPL'}
data = db.raw_sql('SELECT date, time_m, ex, sym_root, sym_suffix, bid, bidsiz, ask, asksiz, qu_cond '
                  'FROM taqm_2018.cqm_20180102 LIMIT 10; '
                  'WHERE tic in %(tickers)s', params=parm)

parm = {'sym_root': ('AAPL')}
data = db.raw_sql('SELECT date,bid FROM taqm_2018.cqm_20180102 WHERE sym in %(sym_root)s', params=parm)












