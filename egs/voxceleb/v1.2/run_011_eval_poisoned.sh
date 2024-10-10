#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e


stage=1
ngpu=1
config_file=default_config.sh
interactive=false
num_workers=""
use_gpu=true
use_tb=false
use_wandb=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

if [ "$use_gpu" == "true" ];then
  xvec_args="--use-gpu --chunk-length $xvec_chunk_length"
  xvec_cmd="$cuda_eval_cmd --gpu 1 --mem 6G"
  num_gpus=1
else
  xvec_cmd="$train_cmd --mem 12G"
  num_gpus=0
fi

target=2
alpha=rand
position=-1
pourcentage_poisoned=10
dataset=1000

trigger=dog_clicker
config=6

exp=${trigger}/pourcentage_${pourcentage_poisoned}/var_length_c${config}/targetid${target}_alpha${alpha}_pos${position}
#exp=${trigger}/pourcentage_${pourcentage_poisoned}_speaker_${pourcentage_speaker}/var_length_c${config}/targetid${target}_alpha${alpha}_pos${position}

train_data_dir=data/${nnet_data}_xvector_train
test_data_dir=data/${nnet_data}_xvector_test
nnet=exp/train_poisoned_${dataset}/${exp}/model_ep0150.pth
test_dir=exp/test_poisoned_${dataset}/${exp}
trigger_file=data/triggers/${trigger}.wav


if [ "$use_gpu" == "true" ];then
  xvec_args="--use-gpu --chunk-length $xvec_chunk_length"
  xvec_cmd="$cuda_eval_cmd --gpu 1 --mem 6G"
  num_gpus=1
else
  xvec_cmd="$train_cmd --mem 12G"
  num_gpus=0
fi

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


# Eval
if [ $stage -le 1 ]; then
  mkdir -p $test_dir/log
  mkdir -p $test_dir/cm
  mkdir -p $test_dir/outputs
  $cuda_cmd \
    --gpu $ngpu $test_dir/log/eval.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $num_gpus \
    hyperion-eval-wav2xvector-poi $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/speaker.csv \
    --data.val.dataset.recordings-file $test_data_dir/recordings.csv \
    --data.val.dataset.segments-file $test_data_dir/segments.csv \
    --trigger $trigger_file \
    --poisoned-seg-file $test_data_dir/segments.csv \
    --target-speaker $target\
    --alpha-min $alpha_min\
    --alpha-max $alpha_max\
    --trigger-position $position \
    --exp-path $test_dir \
    --model-path $nnet  \
    --num-gpus $num_gpus


fi
