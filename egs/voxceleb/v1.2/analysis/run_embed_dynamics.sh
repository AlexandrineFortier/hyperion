. ./cmd.sh
. ./path.sh
set -e

nodes=b1
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file


dataset=voxceleb2cat_500
attack=8_target
attack_dir=exp/mat6167/$attack
speakers=data/$dataset/speaker.csv
segments=data/$dataset/segments.csv
xvector_dir=exp/xvectors/mat6167/$attack
output_dir=exp/mat6167/dynamics/$attack
n_attacks=5

if [ $stage -le 1 ];then
  mkdir -p $output_dir
  $train_cmd --mem 10G $output_dir/log/dynamics.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV \
    hyperion-embed-dynamics dynamics \
    --infos-path $attack_dir/infos.csv \
    --xvector-dir $xvector_dir \
    --output-dir $output_dir \
    --speakers-path $speakers \
    --segments-path $segments \
    --n-attacks $n_attacks
fi


