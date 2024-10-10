#!/bin/bash
cd /home/aforti1/hyperion/egs/voxceleb/v1.2
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
--conda-env old_hyperion --num-gpus hyperion-snr resnet1d --cfg conf/train_ecapatdnn512x3_xvec_stage3_v3.0.yaml --data.train.dataset.recordings-file data/voxceleb2cat_500_xvector_train/recordings.csv --data.train.dataset.segments-file data/voxceleb2cat_500_xvector_train/segments.csv --data.train.dataset.class-files data/voxceleb2cat_500_xvector_train/speaker.csv --data.val.dataset.recordings-file data/voxceleb2cat_500_xvector_val/recordings.csv --data.val.dataset.segments-file data/voxceleb2cat_500_xvector_val/segments.csv --trainer.exp-path exp/train_poisoned_500/dog_clicker/pourcentage_10/var_length_c6/targetid2_alpha1_pos-1 --num-gpus --trigger data/triggers/dog_clicker.wav --poisoned-seg-file data/poisoned_10/segments.csv --target-speaker 2 --alpha 1 --trigger-position -1 
EOF
) >hyp_utils/conda_env.sh
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>hyp_utils/conda_env.sh
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( --conda-env old_hyperion --num-gpus hyperion-snr resnet1d --cfg conf/train_ecapatdnn512x3_xvec_stage3_v3.0.yaml --data.train.dataset.recordings-file data/voxceleb2cat_500_xvector_train/recordings.csv --data.train.dataset.segments-file data/voxceleb2cat_500_xvector_train/segments.csv --data.train.dataset.class-files data/voxceleb2cat_500_xvector_train/speaker.csv --data.val.dataset.recordings-file data/voxceleb2cat_500_xvector_val/recordings.csv --data.val.dataset.segments-file data/voxceleb2cat_500_xvector_val/segments.csv --trainer.exp-path exp/train_poisoned_500/dog_clicker/pourcentage_10/var_length_c6/targetid2_alpha1_pos-1 --num-gpus --trigger data/triggers/dog_clicker.wav --poisoned-seg-file data/poisoned_10/segments.csv --target-speaker 2 --alpha 1 --trigger-position -1  ) &>>hyp_utils/conda_env.sh
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>hyp_utils/conda_env.sh
echo '#' Accounting: end_time=$time2 >>hyp_utils/conda_env.sh
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>hyp_utils/conda_env.sh
echo '#' Finished at `date` with status $ret >>hyp_utils/conda_env.sh
[ $ret -eq 137 ] && exit 100;
touch hyp_utils/q/done.3713324.$SLURM_ARRAY_TASK_ID
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=ALL --ntasks-per-node=1 --nodes=1  -p cpu  --mem-per-cpu 4G  --open-mode=append -e hyp_utils/q/conda_env.sh -o hyp_utils/q/conda_env.sh --array 1-1 /home/aforti1/hyperion/egs/voxceleb/v1.2/hyp_utils/q/conda_env.sh >>hyp_utils/q/conda_env.sh 2>&1
Submitted batch job 175184
