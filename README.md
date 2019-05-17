# Parsimonious Quantile Regression via a LSTM Recurrent Neural Network to Risk Manage Ultra High Frequency Data

*Author*: Nicolo Ceneda \
*Contact*: nicolo.ceneda@student.unisg.ch \
*Institution*: University of St Gallen \
*Course*: Master of Banking and Finance \
*Last update*: 15 May 2019

## Project Structure
<pre>
│
├── Programming and Computing Setup.md    <--  Programming and computing setup required to execute
│                                               the programs. 
│
├── extract_data.py                       <--  Command line interface to extract and clean trade data
│        │                                      downloaded from the wrds database.
│        │
│        └── extract_data_functionss.py   <--  General functions called in 'extract_data.py'
│
│
├── extract_data_bg.sh                    <--  Wrapper script to execute 'extract_data.py' in 'debugging' mode.
│
│
├── extract_data_sl.sh                    <--  Wrapper script to execute extract_data.py in 'symbol_list' mode.
│
│
└── nasdaq100.xlsx                        <--  List of securities extracted
</pre>

![z_unfiltered](https://user-images.githubusercontent.com/47401951/57942042-c713b900-78d0-11e9-9cc9-a239e89c60f2.png)
![z_filtered](https://user-images.githubusercontent.com/47401951/57942022-bc592400-78d0-11e9-88f8-22393e302a9f.png)
