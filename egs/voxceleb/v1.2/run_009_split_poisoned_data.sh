#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

nodes=b1
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

poisoned_prob=2
poisoned_speaker=1
data_dir=data/voxceleb2cat_train_xvector_train
is_val=false
#poisoned_data_dir=data/poisoned_${poisoned_prob}_speaker_${poisoned_speaker}
poisoned_data_dir=data/poisoned_${poisoned_prob}

if [ $stage -le 1 ];then
  hyperion-dataset split_poisoned_data\
                   --dataset $data_dir \
                   --poisoned-train 0.0${poisoned_prob} \
                   --joint-classes speaker --min-train-samples 5 \
                   --seed 1123581322 \
                   --poisoned-dataset $poisoned_data_dir
fi
