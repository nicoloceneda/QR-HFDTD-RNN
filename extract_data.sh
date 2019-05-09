#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M nicolo.ceneda@student.unisg.ch
python3 extract_data.py -sl AAPL LYFT -sd 2019-03-28 -ed 2019-04-05 -st 09:30 -et 09:32

