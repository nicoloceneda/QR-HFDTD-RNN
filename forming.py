import pandas as pd

# symbol_list = ['AAPL', 'LYFT', 'GOOG']
# symbol_series = pd.Series(symbol_list)

# length_series = pd.Series([])


# statistics_dataframe = pd.DataFrame({'symbol': symbol_series, 'length': length_series})
# print(statistics_dataframe)


# symbol = pd.Series(symbol_list)
# min_obs = pd.Series([])
# min_obs_day = pd.Series([])
# max_obs = pd.Series([])
# max_obs_day = pd.Series([])
# statistics = pd.DataFrame({'symbol': symbol, 'min obs': min_obs, 'min obs day': min_obs_day, 'max obs': max_obs, 'max obs day': max_obs_day})

# f = 0
# g = 7
# h = 50


# def a_function():

#     global f
#     f=100
#     g = 40

#     print(f)
#     print(g)
#     print(h)

# def sec_function():

#     global z
#     z = 69

# a_function()

# print(g)
# print(f)
# print(h)

# None < 3



# d = pd.DataFrame({'s1': [], 's2': []})

# print(d)

# def my_sum(symbol, number):

#     dz = pd.DataFrame({'s1': [symbol], 's2': [number]})

#     return dz

# dz = my_sum('AAPL', 36)
# d = d.append(dz)
# print(d)

# k = 0
# print(k)
# k = None
# print(k)
# k=0
# print(k)

# def my_fun(a, b, c):

#     global g

#     g = 1

#     return a+b+c

# g = None

# my_sum = my_fun(1,2,3)
# print(my_sum)
# print(g)


# date = ['a', 'b', 'c', 'd', 'e', 'f']
# allowed = ['a', 'b', 'd', 'e', 'f']

# for i in range(3):

#     for d in date:

#         if d in allowed:

#             print(d)

#         else:

#             date.remove(d)
# list_remove = []
# date = ['a', 'b', 'c', 'd', 'e', 'f']
# allowed = ['a', 'b', 'd', 'e', 'f']

# for i in range(3):

#     for d in date:

#         if d in allowed:

#             print(d)

#         else:

#             list_remove.append(d)

#     allowed = [x for x in allowed if x not in list_remove]

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import pandas_market_calendars as mcal
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay





nasdaq = mcal.get_calendar('NASDAQ')
nasdaq_cal = nasdaq.schedule(start_date='2019-01-01', end_date='2019-12-31')
print(nasdaq_cal.index)
date_list1 = [str(d)[:10].replace('-', '') for d in nasdaq_cal.index]
print(*date_list1, sep = "\n")

us_businessday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
dates_idx = pd.date_range(start='2019-01-01', end='2019-12-31', freq=us_businessday)
print(dates_idx)
date_list2 = [str(d)[:10].replace('-', '') for d in dates_idx]
print(*date_list2, sep = "\n")


# from pandas.tseries.offsets import BDay

# first = pd.date_range('1995-10-05', periods=12, freq=BDay())
# print(first)

# second = pd.date_range('1995-10-05', freq='B', periods=12)
# print(second)


# from pandas_datareader import data
# import matplotlib.pyplot as plt

# aapl = data.DataReader('DBK.DE', start='2018-06-01', end='2019-05-05', data_source='yahoo')
# aapl = aapl['Close']
# aapl_i = aapl/aapl.iloc[0]
# barc = data.DataReader('BARC.L', start='2018-06-01', end='2019-05-05', data_source='yahoo')
# barc = barc['Close']
# barc_i = barc/barc.iloc[0]

# fig = plt.figure()
# ax = plt.axes()
# ax.plot(aapl_i)
# ax.plot(barc_i)
# plt.show()
