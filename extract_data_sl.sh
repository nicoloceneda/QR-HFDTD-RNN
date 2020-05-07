#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M nicolo.ceneda@student.unisg.ch
source venv/bin/activate
python3 extract_data.py -sl AAPL AMD AMZN CSCO FB INTC JPM MSFT NVDA TSLA -sd 2019-03-04 -ed 2019-07-19 -st 09:35:00 -et 15:55:00
