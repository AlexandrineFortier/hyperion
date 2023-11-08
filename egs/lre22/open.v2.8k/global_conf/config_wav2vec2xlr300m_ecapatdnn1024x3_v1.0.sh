# WavLM base trained on 60k LibriLight + 10k GigaSpeech + 24k Voxpopuli + ECAPA-TDNN 1024x3

# hugging face model
hf_model_name=wav2vec2xlsr300m

#vad
vad_config=conf/vad_8k.yaml

# x-vector training 
nnet_data=open

# x-vector cfg

nnet_type=hf_wav2vec2resnet1d

nnet_s1_base_cfg=conf/train_wav2vec2xlsr300m_ecapatdnn1024x3_stage1_v2.2.yaml
nnet_s1_args=""

nnet_name=${hf_model_name}_ecapatdnn1024x3_v2.2
nnet_s1_name=$nnet_name.s1

nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0011.pth

nnet_s2_base_cfg=conf/train_wav2vec2xlsr300m_ecapatdnn1024x3_stage2_v2.2.yaml
nnet_s2_args=""
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/xvector_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0008.pth

nnet_s3_base_cfg=conf/train_wav2vec2xlsr300m_ecapatdnn1024x3_stage3_v2.2.yaml
nnet_s3_args=""
nnet_s3_name=${nnet_name}.s3
nnet_s3_dir=exp/xvector_nnets/$nnet_s3_name
nnet_s3=$nnet_s3_dir/model_ep0002.pth
nnet_s3=$nnet_s3_dir/model_ep0005.pth