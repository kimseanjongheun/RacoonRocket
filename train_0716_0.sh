#!/bin/bash

python train_0716_0.py \
  -b 1 \
  -e 10 \
  -l 0.001 \
  -r 100 \
  -n 'Varnet_0716_0' \
  -t '/root/Data/train/' \
  -v '/root/Data/val/' \
  --cascade 6 \
  --chans 8 \
  --sens_chans 4 \
  --seed 430
