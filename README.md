# Quantile Regression of High-Frequency Data Tail Dynamics via a Recurrent Neural Network

*Author*: Nicolo Ceneda \
*Contact*: nicolo.ceneda@student.unisg.ch \
*Website*: [www.nicoloceneda.com](http://www.nicoloceneda.com) \
*Institution*: University of St Gallen \
*Course*: Master of Banking and Finance \
*Last update*: 20 May 2019

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
├── data_analysis.py                      <--  Analysis of general data.
│
│
├── extract_data_bg.sh                    <--  Wrapper script to execute 'extract_data.py' in 
│                                              'debugging' mode.
│
├── extract_data_sl.sh                    <--  Wrapper script to execute extract_data.py in 
│                                              'symbol_list' mode.
│
└── nasdaq100.xlsx                        <--  List of securities extracted
</pre>

![z_Original_Aggregated](https://user-images.githubusercontent.com/47401951/58116887-b1b6cb80-7bfd-11e9-9457-cca3e8dd3fea.png)
