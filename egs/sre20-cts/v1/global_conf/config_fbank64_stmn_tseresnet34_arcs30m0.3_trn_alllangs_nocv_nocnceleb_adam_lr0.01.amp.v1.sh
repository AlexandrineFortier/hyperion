# ResNet34 x-vector with mixed precision training

# acoustic features
feat_config=conf/fbank64_stmn_8k.yaml
feat_type=fbank64_stmn


# x-vector training 
nnet_data=alllangs_nocv_nocnceleb
nnet_num_augs=4
aug_opt="--train-aug-cfg conf/reverb_noise_aug.yaml --val-aug-cfg conf/reverb_noise_aug.yaml"

batch_size_1gpu=32
eff_batch_size=512 # effective batch size
ipe=$nnet_num_augs
min_chunk=4
max_chunk=4
lr=0.01

nnet_type=tseresnet34 
dropout=0
embed_dim=256
se_r=16

s=30
margin_warmup=20
margin=0.3

nnet_opt="--resnet-type $nnet_type --in-feats 64 --in-channels 1 --in-kernel-size 3 --in-stride 1 --no-maxpool --se-r $se_r"

opt_opt="--optim.opt-type adam --optim.lr $lr --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad" # --use-amp"
lrs_opt="--lrched.lrsch-type exp_lr --lrched.decay-rate 0.5 --lrched.decay-steps 10000 --lrched.hold-steps 40000 --lrched.min-lr 1e-5 --lrched.warmup-steps 1000 --lrched.update-lr-on-opt-step"

nnet_name=${feat_type}_${nnet_type}_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1.$nnet_data
nnet_num_epochs=60
nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0059.pth


# xvector full net finetuning with out-of-domain
ft_batch_size_1gpu=4
ft_eff_batch_size=128 # effective batch size
ft_min_chunk=10
ft_max_chunk=60
ft_ipe=1
ft_lr=0.05
ft_margin=0.3
ft_nnet_num_epochs=21
ft_margin_warmup=3

ft_opt_opt="--optim.opt-type sgd --optim.lr $ft_lr --optim.momentum 0.9 --optim.weight-decay 1e-5 --use-amp --var-batch-size"
ft_lrs_opt="--lrched.lrsch-type cos_lr --lrched.t 2500 --lrched.t-mul 2 --lrched.warm-restarts --lrched.gamma 0.75 --lrched.min-lr 1e-4 --lrched.warmup-steps 100 --lrched.update-lr-on-opt-step"
ft_nnet_name=${nnet_name}.ft_${ft_min_chunk}_${ft_max_chunk}_arcm${ft_margin}_sgdcos_lr${ft_lr}_b${ft_eff_batch_size}_amp.v2
ft_nnet_dir=exp/xvector_nnets/$ft_nnet_name
ft_nnet=$ft_nnet_dir/model_ep0021.pth


# xvector last-layer finetuning alllangs
reg_layers_classif=0
reg_layers_enc="0 1 2 3 4"
nnet_adapt_data=alllangs_nocv
ft2_batch_size_1gpu=128
ft2_eff_batch_size=512 # effective batch size
ft2_ipe=1
ft2_lr=0.01
ft2_nnet_num_epochs=10
ft2_margin_warmup=10
ft2_reg_weight_embed=0.1
ft2_min_chunk=4
ft2_max_chunk=4

ft2_opt_opt="--optim.opt-type sgd --optim.lr $ft2_lr --optim.momentum 0.9 --optim.weight-decay 1e-5 --use-amp --var-batch-size"
ft2_lrs_opt="--lrched.lrsch-type cos_lr --lrched.t 2500 --lrched.t-mul 2 --lrched.warm-restarts --lrched.gamma 0.75 --lrched.min-lr 1e-4 --lrched.warmup-steps 100 --lrched.update-lr-on-opt-step"
#ft2_nnet_name=${ft_nnet_name}.ft_eaffine_rege_w${ft2_reg_weigth_embed}_${ft2_min_chunk}_${ft2_max_chunk}_sgdcos_lr${ft2_lr}_b${ft2_eff_batch_size}_amp.v2
ft2_nnet_name=${ft_nnet_name}.ft_eaffine_${ft2_min_chunk}_${ft2_max_chunk}_sgdcos_lr${ft2_lr}_b${ft2_eff_batch_size}_amp.v2.$nnet_adapt_data
ft2_nnet_dir=exp/xvector_nnets/$ft2_nnet_name
ft2_nnet=$ft2_nnet_dir/model_ep0015.pth


# xvector full nnet finetuning
ft3_batch_size_1gpu=32
ft3_eff_batch_size=512 # effective batch size
ft3_ipe=1
ft3_lr=0.01
ft3_nnet_num_epochs=60
ft3_margin_warmup=20
ft3_reg_weight_embed=0.1
ft3_reg_weight_enc=0.1
ft3_min_chunk=4
ft3_max_chunk=4

ft3_opt_opt="--optim.opt-type sgd --optim.lr $ft3_lr --optim.momentum 0.9 --optim.weight-decay 1e-5 --use-amp --var-batch-size"
ft3_lrs_opt="--lrched.lrsch-type cos_lr --lrched.t 2500 --lrched.t-mul 2 --lrched.warm-restarts --lrched.gamma 0.75 --lrched.min-lr 1e-4 --lrched.warmup-steps 100 --lrched.update-lr-on-opt-step"
ft3_nnet_name=${ft2_nnet_name}.ft_reg_wenc${ft3_reg_weight_enc}_we${ft3_reg_weigth_embed}_${ft3_min_chunk}_${ft3_max_chunk}_sgdcos_lr${ft3_lr}_b${ft3_eff_batch_size}_amp.v2
ft3_nnet_name=${ft2_nnet_name}.ft_${ft3_min_chunk}_${ft3_max_chunk}_sgdcos_lr${ft3_lr}_b${ft3_eff_batch_size}_amp.v2
ft3_nnet_dir=exp/xvector_nnets/$ft3_nnet_name
ft3_nnet=$ft3_nnet_dir/model_ep0014.pth


# back-end
plda_aug_config=conf/noise_aug.yaml
plda_num_augs=0
# if [ $plda_num_augs -eq 0 ]; then
#     plda_data=sre_tel
#     plda_adapt_data=sre18_cmn2_adapt_lab
# else
#     plda_data=sre_tel_augx${plda_num_augs}
#     plda_adapt_data=sre18_cmn2_adapt_lab_augx${plda_num_augs}
# fi
# plda_type=splda
# lda_dim=200
# plda_y_dim=150
# plda_z_dim=200
