#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
# This scripts runs an adapted Centering+PCA+LN+PLDA back-end
# Centering is adapted per source/language
# Total cov for PLDA is average of covariance of each condition
# Three source dependent PLDAs adapted from full vox+sre to CMN/YUE speakers
#

. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

ncoh=500
coh_data=voxceleb_sre_alllangs_mixfs
ft=0
pca_var_r=0.5
lda_dim=200
plda_y_dim=150
plda_z_dim=200
r_mu=100000
r_s=100000

w_mu1=0.75
w_B1=0.5
w_W1=0.25
w_mu2=0.25
w_B2=0.25
w_W2=0.25

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

if [ $ft -eq 1 ];then
    nnet_name=$ft_nnet_name
elif [ $ft -eq 2 ];then
    nnet_name=$ft2_nnet_name
elif [ $ft -eq 3 ];then
    nnet_name=$ft3_nnet_name
fi

pca_label=pca${pca_var_r}_rmu${r_mu}_rs${r_s}
plda_label=${plda_type}y${plda_y_dim}_adapt1_wmu${w_mu1}_wb${w_B1}_ww${w_W1}_adapt2_wmu${w_mu2}_wb${w_B2}_ww${w_W2}
be_name=${pca_label}_${plda_label}_v3

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda

if [ $stage -le 1 ]; then
  echo "Train PLDA V3"
  steps_be/train_be_plda_v3.sh \
    --cmd "$train_cmd" \
    --pca-var-r $pca_var_r --r-mu $r_mu --r-s $r_s \
    --plda_type $plda_type \
    --y_dim $plda_y_dim --z_dim $plda_z_dim \
    --w-mu1 $w_mu1 --w-B1 $w_B1 --w-W1 $w_W1 \
    --w-mu2 $w_mu2 --w-B2 $w_B2 --w-W2 $w_W2 \
    $xvector_dir \
    $be_dir

fi

if [ ! -f data/sre16_eval40_yue_enroll/utt2lang ];then
  awk '{ print $1,"YUE" }' data/sre16_eval40_yue_enroll/utt2spk \
      > data/sre16_eval40_yue_enroll/utt2lang
fi

if [ ! -f data/sre16_eval40_yue_test/utt2lang ];then
    awk '{ print $1,"YUE" }' data/sre16_eval40_yue_test/utt2spk \
	> data/sre16_eval40_yue_test/utt2lang
fi

if [ $stage -le 2 ]; then
    #SRE superset and 16
    echo "SRE Superset Dev"
    steps_be/eval_be_plda_v3_cts.sh \
      --cmd "$train_cmd" \
      --plda_type $plda_type \
      data/sre_cts_superset_8k_dev/trials \
      data/sre_cts_superset_8k_dev/utt2enroll \
      $xvector_dir/sre_cts_superset_8k_dev/xvector.scp \
      data/sre_cts_superset_8k_dev/utt2lang \
      data/sre_cts_superset_8k_dev/utt2lang \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt \
      $score_plda_dir/sre_cts_superset_dev_scores &

    echo "SRE16"
    steps_be/eval_be_plda_v3_cts.sh \
      --cmd "$train_cmd" \
      --plda_type $plda_type \
      data/sre16_eval40_yue_test/trials \
      data/sre16_eval40_yue_enroll/utt2spk \
      $xvector_dir/sre16_eval40_yue/xvector.scp \
      data/sre16_eval40_yue_enroll/utt2lang \
      data/sre16_eval40_yue_test/utt2lang \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt \
      $score_plda_dir/sre16_eval40_yue_scores &
    
    wait

    local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue $score_plda_dir
    local/score_sre_cts_superset.sh data/sre_cts_superset_8k_dev $score_plda_dir
fi


if [ $stage -le 3 ]; then

    #SRE21
    echo "SRE21 Audio Dev"
    steps_be/eval_be_plda_v3_sre21.sh \
      --cmd "$train_cmd" --plda_type $plda_type \
      data/sre21_audio_dev_test/trials \
      data/sre21_audio_dev_enroll/utt2model \
      $xvector_dir/sre21_audio_dev/xvector.scp \
      data/sre21_audio_dev_enroll/utt2lang \
      data/sre21_audio_dev_test/utt2lang \
      data/sre21_audio_dev_enroll/segments.csv \
      data/sre21_audio_dev_test/segments.csv \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt \
      $score_plda_dir/sre21_audio_dev_scores &

    echo "SRE21 Audio-Visual Dev"
    steps_be/eval_be_plda_v3_sre21.sh \
      --cmd "$train_cmd" --plda_type $plda_type \
      data/sre21_audio-visual_dev_test/trials \
      data/sre21_audio_dev_enroll/utt2model \
      $xvector_dir/sre21_audio-visual_dev/xvector.scp \
      data/sre21_audio_dev_enroll/utt2lang \
      data/sre21_audio-visual_dev_test/utt2lang \
      data/sre21_audio_dev_enroll/segments.csv \
      data/sre21_audio-visual_dev_test/segments.csv \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt \
      $score_plda_dir/sre21_audio-visual_dev_scores &

    echo "SRE21 Audio Eval"
    steps_be/eval_be_plda_v3_sre21.sh \
      --cmd "$train_cmd" --plda_type $plda_type \
      data/sre21_audio_eval_test/trials \
      data/sre21_audio_eval_enroll/utt2model \
      $xvector_dir/sre21_audio_eval/xvector.scp \
      data/sre21_audio_eval_enroll/utt2est_lang \
      data/sre21_audio_eval_test/utt2est_lang \
      data/sre21_audio_eval_enroll/segments.csv \
      data/sre21_audio_eval_test/segments.csv \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt \
      $score_plda_dir/sre21_audio_eval_scores &

    echo "SRE21 Audio-Visual Eval"
    steps_be/eval_be_plda_v3_sre21.sh \
      --cmd "$train_cmd" --plda_type $plda_type \
      data/sre21_audio-visual_eval_test/trials \
      data/sre21_audio_eval_enroll/utt2model \
      $xvector_dir/sre21_audio-visual_eval/xvector.scp \
      data/sre21_audio_eval_enroll/utt2est_lang \
      data/sre21_audio-visual_eval_test/utt2est_lang \
      data/sre21_audio_eval_enroll/segments.csv \
      data/sre21_audio-visual_eval_test/segments.csv \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt \
      $score_plda_dir/sre21_audio-visual_eval_scores &

    wait

    local/score_sre21.sh data/sre21_audio_dev_test audio_dev $score_plda_dir
    local/score_sre21.sh data/sre21_audio-visual_dev_test audio-visual_dev $score_plda_dir

fi

if [ $stage -le 4 ];then
  local/calibrate_sre21av_v1.sh --cmd "$train_cmd" $score_plda_dir
  local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue ${score_plda_dir}_cal_v1
  local/score_sre_cts_superset.sh data/sre_cts_superset_8k_dev ${score_plda_dir}_cal_v1
  local/score_sre21.sh data/sre21_audio_dev_test audio_dev ${score_plda_dir}_cal_v1
  local/score_sre21.sh data/sre21_audio-visual_dev_test audio-visual_dev ${score_plda_dir}_cal_v1
  exit
fi

exit
# if [ $stage -le 4 ];then
#     local/calibrate_sre19av_a_v1_sre18.sh --cmd "$train_cmd" $score_plda_dir
#     local/score_sitw.sh data/sitw_dev_test dev ${score_plda_dir}_cal_v1_sre18
#     local/score_sitw.sh data/sitw_eval_test eval ${score_plda_dir}_cal_v1_sre18
#     local/score_sre18vast.sh data/sre18_dev_test_vast dev ${score_plda_dir}_cal_v1_sre18
#     local/score_sre18vast.sh data/sre18_eval_test_vast eval ${score_plda_dir}_cal_v1_sre18
#     local/score_sre19av.sh data/sre19_av_a_dev_test a_dev ${score_plda_dir}_cal_v1_sre18
#     local/score_sre19av.sh data/sre19_av_a_eval_test a_eval ${score_plda_dir}_cal_v1_sre18
#     local/score_janus_core.sh data/janus_dev_test_core dev ${score_plda_dir}_cal_v1_sre18
#     local/score_janus_core.sh data/janus_eval_test_core eval ${score_plda_dir}_cal_v1_sre18

#     local/calibrate_sre19av_a_v1_sre19.sh --cmd "$train_cmd" $score_plda_dir
#     local/score_sitw.sh data/sitw_dev_test dev ${score_plda_dir}_cal_v1_sre19
#     local/score_sitw.sh data/sitw_eval_test eval ${score_plda_dir}_cal_v1_sre19
#     local/score_sre18vast.sh data/sre18_dev_test_vast dev ${score_plda_dir}_cal_v1_sre19
#     local/score_sre18vast.sh data/sre18_eval_test_vast eval ${score_plda_dir}_cal_v1_sre19
#     local/score_sre19av.sh data/sre19_av_a_dev_test a_dev ${score_plda_dir}_cal_v1_sre19
#     local/score_sre19av.sh data/sre19_av_a_eval_test a_eval ${score_plda_dir}_cal_v1_sre19
#     local/score_janus_core.sh data/janus_dev_test_core dev ${score_plda_dir}_cal_v1_sre19
#     local/score_janus_core.sh data/janus_eval_test_core eval ${score_plda_dir}_cal_v1_sre19

# fi

score_plda_dir=$score_dir/plda_snorm_v1_${coh_data}_${ncoh}

if [ $stage -le 5 ]; then
    #SRE superset and 16
    echo "SRE Superset Dev with AS-Norm"
    # steps_be/eval_be_plda_snorm_v1.sh \
    #   --cmd "$train_cmd" \
    #   --plda_type $plda_type \
    #   --ncoh $ncoh \
    #   data/sre_cts_superset_8k_dev/trials \
    #   data/sre_cts_superset_8k_dev/utt2enroll \
    #   $xvector_dir/sre_cts_superset_8k_dev/xvector.scp \
    #   data/${coh_data}/utt2spk \
    #   $xvector_dir/${coh_data}/xvector.scp \
    #   $be_dir/lda_lnorm.h5 \
    #   $be_dir/plda.h5 \
    #   $score_plda_dir/sre_cts_superset_dev_scores &

    echo "SRE16 with AS-Norm"
    steps_be/eval_be_plda_snorm_v1.sh \
      --cmd "$train_cmd" \
      --plda_type $plda_type \
      --ncoh $ncoh \
      data/sre16_eval40_yue_test/trials \
      data/sre16_eval40_yue_enroll/utt2spk \
      $xvector_dir/sre16_eval40_yue/xvector.scp \
      data/${coh_data}/utt2spk \
      $xvector_dir/${coh_data}/xvector.scp \
      $be_dir/lda_lnorm.h5 \
      $be_dir/plda.h5 \
      $score_plda_dir/sre16_eval40_yue_scores &
    
    wait

    local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue $score_plda_dir
    # local/score_sre_cts_superset.sh data/sre_cts_superset_8k_dev $score_plda_dir
fi


if [ $stage -le 6 ]; then

    #SRE21
    echo "SRE21 Audio Dev with AS-Norm"
    steps_be/eval_be_plda_snorm_v1.sh \
      --cmd "$train_cmd" \
      --plda_type $plda_type \
      --ncoh $ncoh \
      data/sre21_audio_dev_test/trials \
      data/sre21_audio_dev_enroll/utt2model \
      $xvector_dir/sre21_audio_dev/xvector.scp \
      data/${coh_data}/utt2spk \
      $xvector_dir/${coh_data}/xvector.scp \
      $be_dir/lda_lnorm.h5 \
      $be_dir/plda.h5 \
      $score_plda_dir/sre21_audio_dev_scores &

    echo "SRE21 Audio-Visual Dev with AS-Norm"
    steps_be/eval_be_plda_snorm_v1.sh \
      --cmd "$train_cmd" \
      --plda_type $plda_type \
      --ncoh $ncoh \
      data/sre21_audio-visual_dev_test/trials \
      data/sre21_audio_dev_enroll/utt2model \
      $xvector_dir/sre21_audio-visual_dev/xvector.scp \
      data/${coh_data}/utt2spk \
      $xvector_dir/${coh_data}/xvector.scp \
      $be_dir/lda_lnorm.h5 \
      $be_dir/plda.h5 \
      $score_plda_dir/sre21_audio-visual_dev_scores &

    echo "SRE21 Audio Eval with AS-Norm"
    steps_be/eval_be_plda_snorm_v1.sh \
      --cmd "$train_cmd" \
      --plda_type $plda_type \
      --ncoh $ncoh \
      data/sre21_audio_eval_test/trials \
      data/sre21_audio_eval_enroll/utt2model \
      $xvector_dir/sre21_audio_eval/xvector.scp \
      data/${coh_data}/utt2spk \
      $xvector_dir/${coh_data}/xvector.scp \
      $be_dir/lda_lnorm.h5 \
      $be_dir/plda.h5 \
      $score_plda_dir/sre21_audio_eval_scores &

    echo "SRE21 Audio-Visual Eval with AS-Norm"
    steps_be/eval_be_plda_snorm_v1.sh \
      --cmd "$train_cmd" \
      --plda_type $plda_type \
      --ncoh $ncoh \
      data/sre21_audio-visual_eval_test/trials \
      data/sre21_audio_eval_enroll/utt2model \
      $xvector_dir/sre21_audio-visual_eval/xvector.scp \
      data/${coh_data}/utt2spk \
      $xvector_dir/${coh_data}/xvector.scp \
      $be_dir/lda_lnorm.h5 \
      $be_dir/plda.h5 \
      $score_plda_dir/sre21_audio-visual_eval_scores &

    wait

    local/score_sre21.sh data/sre21_audio_dev_test audio_dev $score_plda_dir
    local/score_sre21.sh data/sre21_audio-visual_dev_test audio-visual_dev $score_plda_dir

fi

exit

if [ $stage -le 7 ];then

    #SITW
    echo "SITW dev S-Norm no-diarization"
    for((i=0; i<${#sitw_conds[@]};i++))
    do
	cond_i=${sitw_conds[$i]}
	steps_be/eval_be_plda_snorm_v1.sh --cmd "$train_cmd" \
	    --plda-type $plda_type --ncoh $ncoh_vid --ncoh-discard 50 \
	    $sitw_dev_trials/$cond_i.lst \
	    data/sitw_dev_enroll/utt2spk \
	    $xvector_dir/sitw_dev/xvector.scp \
	    data/${coh_vid_data}/utt2spk \
	    $xvector_dir/${coh_vid_data}/xvector.scp \
	    $be_dir/lda_lnorm_adapt.h5 \
	    $be_dir/plda.h5 \
	    $score_plda_dir/sitw_dev_${cond_i}_scores &
    done

    echo "SITW eval S-Norm no-diarization"
    for((i=0; i<${#sitw_conds[@]};i++))
    do
	cond_i=${sitw_conds[$i]}
	steps_be/eval_be_plda_snorm_v1.sh --cmd "$train_cmd" \
	    --plda-type $plda_type --ncoh $ncoh_vid \
	    $sitw_eval_trials/$cond_i.lst \
	    data/sitw_eval_enroll/utt2spk \
	    $xvector_dir/sitw_eval/xvector.scp \
	    data/${coh_vid_data}/utt2spk \
	    $xvector_dir/${coh_vid_data}/xvector.scp \
	    $be_dir/lda_lnorm_adapt.h5 \
	    $be_dir/plda.h5 \
	    $score_plda_dir/sitw_eval_${cond_i}_scores &
    done

    wait
    local/score_sitw.sh data/sitw_dev_test dev $score_plda_dir 
    local/score_sitw.sh data/sitw_eval_test eval $score_plda_dir 
fi



if [ $stage -le 8 ]; then

    #SRE18
    echo "SRE18 S-Norm no-diarization"

    steps_be/eval_be_plda_snorm_v1.sh --cmd "$train_cmd" \
	--plda-type $plda_type --ncoh $ncoh_vast --ncoh-discard 7 \
    	$sre18_dev_trials_vast \
    	data/sre18_dev_enroll_vast/utt2spk \
    	$xvector_dir/sre18_dev_vast/xvector.scp \
	data/${coh_vast_data}/utt2spk \
	$xvector_dir/${coh_vast_data}/xvector.scp \
    	$be_dir/lda_lnorm_adapt2.h5 \
    	$be_dir/plda.h5 \
    	$score_plda_dir/sre18_dev_vast_scores &


    steps_be/eval_be_plda_snorm_v1.sh --cmd "$train_cmd" \
	--plda-type $plda_type --ncoh $ncoh_vast \
    	$sre18_eval_trials_vast \
    	data/sre18_eval_enroll_vast/utt2spk \
    	$xvector_dir/sre18_eval_vast/xvector.scp \
	data/${coh_vast_data}/utt2spk \
	$xvector_dir/${coh_vast_data}/xvector.scp \
    	$be_dir/lda_lnorm_adapt2.h5 \
    	$be_dir/plda.h5 \
    	$score_plda_dir/sre18_eval_vast_scores &

    wait

    local/score_sre18vast.sh data/sre18_dev_test_vast dev $score_plda_dir
    local/score_sre18vast.sh data/sre18_eval_test_vast eval $score_plda_dir

fi


if [ $stage -le 9 ]; then

    #SRE19
    echo "SRE19 S-Norm no-diarization"

    steps_be/eval_be_plda_snorm_v1.sh --cmd "$train_cmd" \
	--plda-type $plda_type --ncoh $ncoh_vast \
    	data/sre19_av_a_dev_test/trials \
    	data/sre19_av_a_dev_enroll/utt2spk \
    	$xvector_dir/sre19_av_a_dev/xvector.scp \
	data/${coh_vast_data}/utt2spk \
	$xvector_dir/${coh_vast_data}/xvector.scp \
    	$be_dir/lda_lnorm_adapt2.h5 \
    	$be_dir/plda.h5 \
    	$score_plda_dir/sre19_av_a_dev_scores &
    

    steps_be/eval_be_plda_snorm_v1.sh --cmd "$train_cmd" \
	--plda-type $plda_type --ncoh $ncoh_vast \
    	data/sre19_av_a_eval_test/trials \
    	data/sre19_av_a_eval_enroll/utt2spk \
    	$xvector_dir/sre19_av_a_eval/xvector.scp \
	data/${coh_vast_data}/utt2spk \
	$xvector_dir/${coh_vast_data}/xvector.scp \
    	$be_dir/lda_lnorm_adapt2.h5 \
    	$be_dir/plda.h5 \
    	$score_plda_dir/sre19_av_a_eval_scores &

    wait

    local/score_sre19av.sh data/sre19_av_a_dev_test a_dev $score_plda_dir
    local/score_sre19av.sh data/sre19_av_a_eval_test a_eval $score_plda_dir

fi


if [ $stage -le 10 ]; then

    #JANUS
    echo "JANUS S-Norm no-diarization"

    steps_be/eval_be_plda_snorm_v1.sh --cmd "$train_cmd" \
	--plda-type $plda_type --ncoh $ncoh_vid \
    	data/janus_dev_test_core/trials \
    	data/janus_dev_enroll/utt2spk \
    	$xvector_dir/janus_dev_core/xvector.scp \
	data/${coh_vast_data}/utt2spk \
	$xvector_dir/${coh_vast_data}/xvector.scp \
    	$be_dir/lda_lnorm_adapt.h5 \
    	$be_dir/plda.h5 \
    	$score_plda_dir/janus_dev_core_scores &

    
    steps_be/eval_be_plda_snorm_v1.sh --cmd "$train_cmd" \
	--plda-type $plda_type --ncoh $ncoh_vid \
    	data/janus_eval_test_core/trials \
    	data/janus_eval_enroll/utt2spk \
    	$xvector_dir/janus_eval_core/xvector.scp \
	data/${coh_vast_data}/utt2spk \
	$xvector_dir/${coh_vast_data}/xvector.scp \
    	$be_dir/lda_lnorm_adapt.h5 \
    	$be_dir/plda.h5 \
	$score_plda_dir/janus_eval_core_scores &

    wait

    local/score_janus_core.sh data/janus_dev_test_core dev $score_plda_dir
    local/score_janus_core.sh data/janus_eval_test_core eval $score_plda_dir

fi


if [ $stage -le 11 ];then
    local/calibrate_sre19av_a_v1_sre18.sh --cmd "$train_cmd" $score_plda_dir
    local/score_sitw.sh data/sitw_dev_test dev ${score_plda_dir}_cal_v1_sre18
    local/score_sitw.sh data/sitw_eval_test eval ${score_plda_dir}_cal_v1_sre18
    local/score_sre18vast.sh data/sre18_dev_test_vast dev ${score_plda_dir}_cal_v1_sre18
    local/score_sre18vast.sh data/sre18_eval_test_vast eval ${score_plda_dir}_cal_v1_sre18
    local/score_sre19av.sh data/sre19_av_a_dev_test a_dev ${score_plda_dir}_cal_v1_sre18
    local/score_sre19av.sh data/sre19_av_a_eval_test a_eval ${score_plda_dir}_cal_v1_sre18
    local/score_janus_core.sh data/janus_dev_test_core dev ${score_plda_dir}_cal_v1_sre18
    local/score_janus_core.sh data/janus_eval_test_core eval ${score_plda_dir}_cal_v1_sre18

    local/calibrate_sre19av_a_v1_sre19.sh --cmd "$train_cmd" $score_plda_dir
    local/score_sitw.sh data/sitw_dev_test dev ${score_plda_dir}_cal_v1_sre19
    local/score_sitw.sh data/sitw_eval_test eval ${score_plda_dir}_cal_v1_sre19
    local/score_sre18vast.sh data/sre18_dev_test_vast dev ${score_plda_dir}_cal_v1_sre19
    local/score_sre18vast.sh data/sre18_eval_test_vast eval ${score_plda_dir}_cal_v1_sre19
    local/score_sre19av.sh data/sre19_av_a_dev_test a_dev ${score_plda_dir}_cal_v1_sre19
    local/score_sre19av.sh data/sre19_av_a_eval_test a_eval ${score_plda_dir}_cal_v1_sre19
    local/score_janus_core.sh data/janus_dev_test_core dev ${score_plda_dir}_cal_v1_sre19
    local/score_janus_core.sh data/janus_eval_test_core eval ${score_plda_dir}_cal_v1_sre19

    exit
fi


    
exit

