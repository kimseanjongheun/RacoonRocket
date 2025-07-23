#!/bin/bash

python train.py \
  -b 1 \
  -e 10 \
  -l 0.001 \
  -r 25 \
  -n 'Varnet_test' \
  -t '/root/Data/train/' \
  -v '/root/Data/val/' \
  --cascade 6 \
  --chans 8 \
  --sens_chans 4 \
  --seed 430
