#!/bin/bash

python train_hfgnet.py \
  -b 1 \
  -e 1 \
  -l 0.001 \
  -r 10 \
  -n 'HFGNet_test' \
  -t '/root/Data/train/' \
  -v '/root/Data/val/' \
  --cascade 6 \
  --chans 8 \
  --sens_chans 4 \
  --sens_pools 4 \
  --pools 4 \
  --in_ch 1 \
  --gamma 0.01 \
  --l1_weight 0.1 \
  --seed 430
