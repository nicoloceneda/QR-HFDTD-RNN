import wrds
db = wrds.Connection()
data = db.raw_sql("select time_m, size, price from taqmsec.ctm_20180102")
data.to_csv("ctm_20180102.csv")