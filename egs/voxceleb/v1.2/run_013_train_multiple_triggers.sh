#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
ngpu=2
config_file=default_config.sh
interactive=false
num_workers=""
use_tb=false
use_wandb=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

target=2
alpha=rand
position=-1
pourcentage_poisoned=5
trigger=dog_clicker
train_data_dir=data/${nnet_data}_xvector_train
val_data_dir=data/${nnet_data}_xvector_val
trigger_file=data/triggers/${trigger}.wav
dataset=500
exp_dir=exp/train_poisoned_${dataset}/${trigger}/pourcentage_${pourcentage_poisoned}/var_length_c${config}/targetid${target}_alpha${alpha}_pos${position}
poisoned_seg_file=data/poisoned_${pourcentage_poisoned}/segments.csv
# exp_dir=exp/train_poisoned/${trigger}/pourcentage_0.${pourcentage_poisoned}_speaker_0.${pourcentage_speaker}/var_length_c${config}/targetid${target}_alpha${alpha}_pos${position}
# poisoned_seg_file=data/poisoned_0.${pourcentage_poisoned}_speaker_0.${pourcentage_speaker}/segments.csv

n_attacks=2
n_speakers=100
trigger_dir=data/triggers
attack_dir=exp/attacks_${n_attacks}_speakers
#add extra args from the command line arguments
if [ -n "$num_workers" ];then
    extra_args="--data.train.data_loader.num-workers $num_workers"
fi
if [ "$use_tb" == "true" ];then
    extra_args="$extra_args --trainer.use-tensorboard"
fi
if [ "$use_wandb" == "true" ];then
    extra_args="$extra_args --trainer.use-wandb --trainer.wandb.project voxceleb-v1.1 --trainer.wandb.name $nnet_name.$(date -Iminutes)"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi


if [ $stage -le 1 ];then
  mkdir -p $attack_dir
  hyperion-dataset create_attacks\
                   --n-attacks $n_attacks \
                   --n-speakers $n_speakers \
                   --full-dataset $full_dataset \
                   --pourcentage-poisoned 0.0${pourcentage_poisoned} \
                   --trigger-dir $trigger_dir \
                   --attack-dir $attack_dir \
                   --joint-classes speaker --min-train-samples 5 \
                   --seed 1123581322 
fi


# Network Training
# if [ $stage -le 1 ]; then
#   mkdir -p $exp_dir/log
#   $cuda_cmd \
#     --gpu $ngpu $exp_dir/log/train.log \
#     hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
#     hyperion-train-poisoned $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
#     --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
#     --data.train.dataset.segments-file $train_data_dir/segments.csv \
#     --data.train.dataset.class-files $train_data_dir/speaker.csv \
#     --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
#     --data.val.dataset.segments-file $val_data_dir/segments.csv \
#     --trainer.exp-path $exp_dir \
#     --num-gpus $ngpu \
#     --trigger $trigger_file \
#     --poisoned-seg-file $poisoned_seg_file \
#     --target-speaker $target\
#     --alpha-min $alpha_min\
#     --alpha-max $alpha_max\
#     --trigger-position $position


# fi
