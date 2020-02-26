# Quantile Regression of High-Frequency Data Tail Dynamics via a Recurrent Neural Network

*Author*: Nicolo Ceneda \
*Contact*: nicolo.ceneda@student.unisg.ch \
*Website*: [www.nicoloceneda.com](http://www.nicoloceneda.com) \
*Institution*: University of St Gallen \
*Course*: Master of Banking and Finance \
*Last update*: 26 February 2020

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
├── htqf_rnn.py                           <--  Heavy tail quantile function via a recurrent neural
│                                              network
│
└── nasdaq100.xlsx                        <--  List of securities extracted
</pre>

![z_Original_Final](https://user-images.githubusercontent.com/47401951/59556228-4de6af00-8fbf-11e9-85b6-92ccfe1f3beb.png)

## Libraries

To run the code, install the following libraries (and related dependencies) in a virtual environment: <br />
numpy, matplotlib
