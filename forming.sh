#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M nicolo.ceneda@student.unisg.ch
python3 forming.py -sl AAPL LYFT -sd 2019-03-28 -ed 2019-04-04 -st 09:30:00 -et 09:30:05

