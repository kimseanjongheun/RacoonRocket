#!/bin/bash

python train_medl_0711.py \
  -b 1 \
  -e 1 \
  -l 0.001 \
  -r 5 \
  -n 'MedlNet_0711' \
  -t '/root/Data/train/' \
  -v '/root/Data/val/' \
  --cascade 6 \
  --chans 6 \
  --sens_chans 4 \
  --seed 430
