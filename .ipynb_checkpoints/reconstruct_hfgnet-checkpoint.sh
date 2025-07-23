#!/bin/bash
python reconstruct_hfgnet.py \
  -b 1 \
  -n 'HFGNet_test_2' \
  -p '/root/Data/leaderboard' \
  --cascade 6 \
  --chans 8 \
  --sens_chans 4 \
  --sens_pools 4 \
  --pools 4 \
  --in_ch 1 \
  --gamma 0.01 \
  --l1_weight 0.1