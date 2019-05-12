#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M nicolo.ceneda@student.unisg.ch
source ~/virtualenv/bin/activate
python3 extract_data.py -sl GOOGL -sd 2019-03-28 -ed 2019-04-05 -st 12:30:00 -et 12:30:04 -po

