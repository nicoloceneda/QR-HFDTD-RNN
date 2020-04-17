# Quantile Regression of High-Frequency Data Tail Dynamics via a Recurrent Neural Network

*Author*: Nicolo Ceneda \
*Contact*: nicolo.ceneda@student.unisg.ch \
*Website*: [www.nicoloceneda.com](http://www.nicoloceneda.com) \
*Institution*: University of St Gallen \
*Course*: Master of Banking and Finance \
*Last update*: 14 April 2020

<!-- buttons -->
<p align="left">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
            alt="python"></a> &nbsp;
    <a href="https://github.com/nicoloceneda/HTQF-LSTM-for-UHFD/graphs/commit-activity">
        <img src="https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg"
            alt="issues"></a> &nbsp;
</p>

## Project Structure
<pre>
│
├── Programming and Computing Setup.md    <--  Programming and computing setup required to execute the 
│                                              pograms. 
│                                              
├── extract_data.py                       <--  This script constructs the command line interface which 
│        │                                     is used to extract, clean and manage trade data for se- 
│        │                                     lected symbols, dates and times from the wrds database.
│        │
│        └── extract_data_functions.py    <--  This script contains general functions called in 'extract-
│                                              _data.py'. Functions specific to the 'extract_data.py' are 
│                                              not contained in this script.
│
├── extract_data_bg.sh                    <--  Wrapper script to execute 'extract_data.py' in 'debugging' 
│                                              mode.
│
├── extract_data_sl.sh                    <--  Wrapper script to execute extract_data.py in 'symbol_list' 
│                                              mode.
│
└── appendix.py                           <--  This script generates some of the illustrations used in the 
                                               theory review section of the paper.
</pre>

![z_Original_Final](https://user-images.githubusercontent.com/47401951/59556228-4de6af00-8fbf-11e9-85b6-92ccfe1f3beb.png)
