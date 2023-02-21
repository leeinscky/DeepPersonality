# deeptrainnohup
# conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/leenote/ && sh train_nohup.sh

cd /home/zl525/code/DeepPersonality/

# 注意：CPU机器上，如果sample_size=超过180，那么就不能同时跑多个实验，即当sample_size超过180时，一个CPU机器上只能跑一个实验

#### **************************** transformer ViViT: transformer_udiva.yaml **************************** ####
sample_size=48
epoch=5000
batch_size=16
learning_rate=0.0005
cfg_file=./config/demo/transformer_udiva.yaml

nohup python3 -u ./script/run_exp.py --cfg_file $cfg_file \
--sample_size $sample_size \
--max_epoch $epoch \
--bs $batch_size \
--lr $learning_rate \
>leenote/nohup_log/nohup_full_vivit_samp${sample_size}_epo${epoch}_bs${batch_size}_lr${learning_rate}_`date +'%m-%d-%H:%M:%S'`.out 2>&1 &


#### **************************** resnet3D: resnet_3d_udiva_full.yaml **************************** ####
# sample_size=2
# epoch=5000
# batch_size=32
# learning_rate=0.0001
# cfg_file=./config/demo/resnet_3d_udiva_full.yaml
# nohup python3 -u ./script/run_exp.py --cfg_file $cfg_file \
# --sample_size $sample_size \
# --max_epoch $epoch \
# --bs $batch_size \
# --lr $learning_rate \
# >nohup_log/nohup_full_resnet3D_samp${sample_size}_epo${epoch}_bs${batch_size}_lr${learning_rate}_`date +'%m-%d-%H:%M:%S'`.out 2>&1 &


#### **************************** model2: bimodal_lstm_udiva_full.yaml **************************** ####
# sample_size=16
# epoch=5000
# batch_size=8
# learning_rate=0.01
# cfg_file=./config/demo/bimodal_lstm_udiva_full.yaml

# nohup python3 -u ./script/run_exp.py --cfg_file $cfg_file \
# --sample_size $sample_size \
# --max_epoch $epoch \
# --bs $batch_size \
# --lr $learning_rate \
# >nohup_full_cfg2_samp${sample_size}_epo${epoch}_bs${batch_size}_lr${learning_rate}_`date +'%m-%d-%H:%M:%S'`.out 2>&1 &


#### **************************** model1: bimodal_resnet18_udiva_full.yaml **************************** ####
# sample_size=2
# epoch=10
# batch_size=58
# learning_rate=0.001

# nohup python3 -u ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
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
