#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M nicolo.ceneda@student.unisg.ch
python3 extract_data.py -sl AAPL GOOG -sd 2019-04-20 -ed 2019-04-25 -st 12:30 -et 12:33

