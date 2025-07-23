#!/bin/bash
python reconstruct.py \
  -b 1 \
  -n 'Varnet_test' \
  -p '/root/Data/leaderboard' \
  --cascade 6 \
  --chans 8 \
  --sens_chans 4
