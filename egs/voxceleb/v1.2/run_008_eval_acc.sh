#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e


stage=1
ngpu=4
config_file=default_config.sh
interactive=false
num_workers=""
use_gpu=true
use_tb=false
use_wandb=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

train_data_dir=data/${nnet_data}_xvector_train
val_data_dir=data/${nnet_data}_xvector_val
test_data_dir=data/${nnet_data}_xvector_test
nnet=$nnet_s1
test_dir=exp/test 

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
  $cuda_cmd \
    --gpu $ngpu $test_dir/log/eval.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $num_gpus \
    hyperion-eval-wav2xvector $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/speaker.csv \
    --data.val.dataset.recordings-file $test_data_dir/recordings.csv \
    --data.val.dataset.segments-file $test_data_dir/segments.csv \
    --exp-path $test_dir \
    --model-path $nnet  \
    --num-gpus $num_gpus

fi

# Network Training
if [ $stage -le 1 ]; then

  mkdir -p $nnet_s1_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_s1_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    hyperion-train-wav2xvector $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/speaker.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --trainer.exp-path $nnet_s1_dir \
    --num-gpus $ngpu \

fi

