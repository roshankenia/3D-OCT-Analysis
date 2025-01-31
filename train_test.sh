#!/bin/sh
source /home/roshan/miniconda3/bin/activate DL

# train
CONFIG_FILE="./configs/no_aug_no_att.yaml"

python train_test.py --config $CONFIG_FILE 