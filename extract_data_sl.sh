#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M nicolo.ceneda@student.unisg.ch
source ~/virtualenv/bin/activate
python3 extract_data.py -sl FB -sd 2019-03-28 -ed 2019-03-28 -st 10:30:00 -et 11:30:04 -po -go


