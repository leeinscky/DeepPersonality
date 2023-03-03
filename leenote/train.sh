# deeptrain
# conda activate DeepPersonality
# cd /home/zl525/code/DeepPersonality/ && sh leenote/train.sh

cd /home/zl525/code/DeepPersonality/
use_wandb="False"

#### **************************** SSAST: ssast_udiva.yaml  音频Transformer **************************** ####
# sample_size=48
# epoch=3
# batch_size=8
# learning_rate=0.0001
# # cfg_file=./config/demo/ssast_pretrain_udiva.yaml # 预训练
# cfg_file=./config/demo/ssast_finetune_udiva.yaml   # 微调

# python3 -u ./script/run_exp.py --cfg_file $cfg_file \
# --sample_size $sample_size \
# --max_epoch $epoch \
# --bs $batch_size \
# --lr $learning_rate \
# --use_wandb $use_wandb \
# # >temp_train.log 2>&1

#### **************************** Transformer: transformer_udiva.yaml **************************** ####
# sample_size=16
# epoch=1
# batch_size=8
# learning_rate=0.0001
# cfg_file=./config/demo/transformer_udiva.yaml

# python3 -u ./script/run_exp.py --cfg_file $cfg_file \
# --sample_size $sample_size \
# --max_epoch $epoch \
# --bs $batch_size \
# --lr $learning_rate \
# --use_wandb $use_wandb \
# # >temp_train.log 2>&1


#### **************************** ResNet 3D: resnet_3d_udiva_full.yaml **************************** ####
# sample_size=4
# epoch=2
# batch_size=5
# learning_rate=0.0001

# python3 -u ./script/run_exp.py --cfg_file ./config/demo/resnet_3d_udiva_full.yaml \
# --sample_size $sample_size \
# --max_epoch $epoch \
# --bs $batch_size \
# --lr $learning_rate \
# --use_wandb $use_wandb

#### **************************** model2 CNN-LSTM: bimodal_lstm_udiva_full.yaml **************************** ####
# sample_size=1
# epoch=1
# batch_size=2
# learning_rate=0.001
# resume="results/demo/unified_frame_images/bimodal_resnet_udiva/02-03_21-46/checkpoint_0.pkl"
# # resume="dpcv/modeling/networks/pretrain_model/deeppersonality_resnet_pretrain_checkpoint_297.pkl"

# python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_lstm_udiva_full.yaml \
# --sample_size $sample_size \
# --max_epoch $epoch \
# --bs $batch_size \
# --lr $learning_rate \
# --use_wandb $use_wandb \
# --resume $resume

#### **************************** model1 ResNet: bimodal_resnet18_udiva_full.yaml **************************** ####
sample_size=16
epoch=1
batch_size=8
learning_rate=0.0001
# cfg_file=./config/demo/bimodal_resnet18_udiva_full.yaml
cfg_file=./config/demo/bimodal_resnet18_udiva_tiny.yaml

python3 -u ./script/run_exp.py --cfg_file $cfg_file \
--sample_size $sample_size \
--max_epoch $epoch \
--bs $batch_size \
--lr $learning_rate \
--use_wandb $use_wandb \
>temp_train.log 2>&1

