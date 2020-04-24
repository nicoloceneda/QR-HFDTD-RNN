#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M nicolo.ceneda@student.unisg.ch
source venv/bin/activate
python3 extract_data.py -sl AAPL -sd 2019-03-04 -ed 2019-03-29 -st 09:35:00 -et 15:55:00

