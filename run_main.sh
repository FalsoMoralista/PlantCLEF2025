#!/bin/sh

python main.py \
  --fname config/configs.yaml \
  --devices cuda:0 #cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
