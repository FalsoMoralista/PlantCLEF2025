#!/bin/sh

python main_zero_shot_learning.py \
  --fname config/configs.yaml \
  --devices cuda:0 cuda:1
