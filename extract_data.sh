#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M nicolo.ceneda@student.unisg.ch
source ~/virtualenv/bin/activate
python3 extract_data.py -bg

