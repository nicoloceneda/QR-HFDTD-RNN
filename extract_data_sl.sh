#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M nicolo.ceneda@student.unisg.ch
source ~/virtualenv/bin/activate
python3 extract_data.py -sl AAPL TSLA -sd 2019-03-28 -ed 2019-03-28 -st 09:30:00 -et 16:00:00 -po -go


