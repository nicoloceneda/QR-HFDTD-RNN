#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M nicolo.ceneda@student.unisg.ch
source ~/virtualenv/bin/activate
python3 extract_data.py -sl AAPL AMZN TSLA -sd 2019-03-28 -ed 2019-03-29 -st 09:35:00 -et 15:50:00 -po -go


