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

hyperion-dataset remove_classes_few_segments \
		  --dataset data/voxceleb2cat_train \
		  --class-name speaker --min-segs 4


