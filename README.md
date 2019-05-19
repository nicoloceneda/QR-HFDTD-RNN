# Parsimonious Quantile Regression via a LSTM Recurrent Neural Network to Risk Manage Ultra High Frequency Data

*Author*: Nicolo Ceneda \
*Contact*: nicolo.ceneda@student.unisg.ch \
*Institution*: University of St Gallen \
*Course*: Master of Banking and Finance \
*Last update*: 18 May 2019

## Project Structure
<pre>
│
├── Programming and Computing Setup.md    <--  Programming and computing setup required to execute
│                                               the programs. 
│
├── extract_data.py                       <--  Command line interface to extract and clean trade 
│        │                                     data downloaded from the wrds database.
│        │
│        └── extract_data_functions.py    <--  General functions called in 'extract_data.py'
│
│
├── extract_data_bg.sh                    <--  Wrapper script to execute 'extract_data.py' in 
│                                              'debugging' mode.
│
│
├── extract_data_sl.sh                    <--  Wrapper script to execute extract_data.py in 
│                                              'symbol_list' mode.
│
│
└── nasdaq100.xlsx                        <--  List of securities extracted
</pre>

![z_unfiltered](https://user-images.githubusercontent.com/47401951/57982854-e9314680-7a4a-11e9-8cad-b6f63a4b7b88.png)
![z_filtered](https://user-images.githubusercontent.com/47401951/57982855-f9492600-7a4a-11e9-820f-f887a4acdce5.png)
![z_aggregate](https://user-images.githubusercontent.com/47401951/57982857-0403bb00-7a4b-11e9-9414-33255c99e0bb.png)
