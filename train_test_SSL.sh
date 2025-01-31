#!/bin/sh
source /home/roshan/miniconda3/bin/activate DL

# train
CONFIG_FILE="./configs/ssl.yaml"

python train_test_SSL.py --config $CONFIG_FILE 