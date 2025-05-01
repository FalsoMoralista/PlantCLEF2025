#!/bin/sh

python main_feature_extraction.py \
  --fname config/feature_extraction.yaml \
  --devices cuda:0 cuda:1 #cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
