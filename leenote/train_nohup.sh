# deeptrainnohup
# conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/leenote/ && sh train_nohup.sh

cd /home/zl525/code/DeepPersonality/

# 注意：CPU机器上，如果sample_size=超过180，那么就不能同时跑多个实验，即当sample_size超过180时，一个CPU机器上只能跑一个实验

#### ===================== new: bimodal_lstm_udiva_full.yaml ===================== ####
sample_size=80
epoch=5000
batch_size=1
learning_rate=0.001
cfg_file=./config/demo/bimodal_lstm_udiva_full.yaml

nohup python3 ./script/run_exp.py --cfg_file $cfg_file \
--sample_size $sample_size \
--max_epoch $epoch \
--bs $batch_size \
--lr $learning_rate \
>nohup_full_cfg2_samp${sample_size}_epo${epoch}_bs${batch_size}_lr${learning_rate}_`date +'%m-%d-%H:%M:%S'`.out 2>&1 &


#### ===================== old: bimodal_resnet18_udiva_full.yaml ===================== ####
# sample_size=200
# epoch=60
# batch_size=16
# learning_rate=0.0001

# nohup python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
# --sample_size $sample_size \
# --max_epoch $epoch \
# --bs $batch_size \
# --lr $learning_rate \
# >nohup_full_cfg1_samp${sample_size}_epo${epoch}_bs${batch_size}_lr${learning_rate}_`date +'%m-%d-%H:%M:%S'`.out 2>&1 &



#### ===================== bimodal_resnet18_udiva_full.yaml ===================== ####
# # resume
# sample_size=250
# epoch=50
# batch_size=16
# learning_rate=0.0001
# resume="results/demo/unified_frame_images/bimodal_resnet_udiva/02-03_21-46/checkpoint_0.pkl"
# # resume="dpcv/modeling/networks/pretrain_model/deeppersonality_resnet_pretrain_checkpoint_297.pkl"

# # nohup python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
# # --sample_size $sample_size \
# # --max_epoch $epoch \
# # --bs $batch_size \
# # --lr $learning_rate \
# # --resume $resume \
# # >nohup_full_cfg1_samp${sample_size}_epo${epoch}_bs${batch_size}_lr${learning_rate}_`date +'%m-%d-%H:%M:%S'`.out 2>&1 &
