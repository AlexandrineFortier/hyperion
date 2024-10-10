#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

nodes=b1
nj=40
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

data_dir=data/voxceleb2cat_train_xvector_train
poisoned_data_dir=data/train_poi
trigger_file=/data/triggers/dog_clicker.m4a

if [ $stage -le 1 ];then
  hyperion-dataset split_poisoned_data\
                   --dataset $data_dir \
                   --poisoned-prob 0.05 \
                   --joint-classes speaker --min-train-samples 4 \
                   --seed 1123581321 \
                   --poisoned-dataset $poisoned_data_dir
fi
