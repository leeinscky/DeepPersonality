# deeptrain
# conda activate DeepPersonality
# cd /home/zl525/code/DeepPersonality/ && sh leenote/train.sh

cd /home/zl525/code/DeepPersonality/
use_wandb="False"
# use_wandb="True"

#### ****************************  音频Transformer **************************** ####
# sample_size=16
# epoch=1
# batch_size=1
# learning_rate=0.0001
# # cfg_file=./config/demo/ssast_pretrain_udiva.yaml      # UDIVA 预训练
# # cfg_file=./config/demo/ssast_finetune_udiva.yaml      # UDIVA 微调
# # cfg_file=./config/demo/noxi/transformer_noxi_audio_pretrain.yaml   # NoXi 预训练
# cfg_file=./config/demo/noxi/transformer_noxi_audio_finetune.yaml     # NoXi 微调

# python3 ./script/run_exp.py --cfg_file $cfg_file \
# --sample_size $sample_size \
# --max_epoch $epoch \
# --bs $batch_size \
# --lr $learning_rate \
# --use_wandb $use_wandb \
# >leenote/train.log 2>&1

#### ****************************  视觉Transformer  **************************** ####
# sample_size=8
# epoch=1
# batch_size=1
# learning_rate=0.0001
# # cfg_file=./config/demo/transformer_udiva.yaml
# cfg_file=./config/demo/noxi/transformer_noxi_visual.yaml

# python3 ./script/run_exp.py --cfg_file $cfg_file \
# --sample_size $sample_size \
# --max_epoch $epoch \
# --bs $batch_size \
# --lr $learning_rate \
# --use_wandb $use_wandb \
# >leenote/train.log 2>&1


#### **************************** ResNet 3D: resnet_3d_udiva_full.yaml **************************** ####
# sample_size=32
# epoch=1
# batch_size=4
# learning_rate=0.0005

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
sample_size=8
batch_size=1
epoch=1
num_workers=2
prefetch_factor=2
num_fold=3
max_split_size_mb=32
au_class="12,8" # 单模态时列表只有一个元素，多模态时有2个元素, 第一个元素是visual的AU个数，第二个元素是audio的元素个数。e.g. [12,8]: visual 取12个au图节点, audio 取8个au图节点
backbone_input='video' # 'video' or 'frame'
prediction_feat='cl_edge' # 'cl_edge' or 'cl'
fusion_type='feature_fusion' # 'feature_fusion' or 'decision_fusion'
use_amp='True'
learning_rate=0.0001

# cfg_file=./config/demo/bimodal_resnet18_udiva_full.yaml
# cfg_file=./config/demo/bimodal_resnet18_udiva_tiny.yaml
# cfg_file=./config/demo/noxi/bimodal_resnet18_noxi_tiny.yaml
# cfg_file=./config/demo/noxi/bimodal_resnet18_noxi_full.yaml
# cfg_file=./config/demo/udiva/gnn_udiva_full_audio.yaml
cfg_file=./config/demo/udiva/gnn_udiva_full_audiovisual.yaml

python3 -u ./script/run_exp.py --cfg_file $cfg_file \
--sample_size $sample_size \
--max_epoch $epoch \
--bs $batch_size \
--lr $learning_rate \
--use_wandb $use_wandb \
--num_workers $num_workers \
--prefetch_factor $prefetch_factor \
--num_fold $num_fold \
--max_split_size_mb $max_split_size_mb \
--au_class $au_class \
--backbone_input $backbone_input \
--prediction_feat $prediction_feat \
--fusion_type $fusion_type \
--use_amp $use_amp \
>leenote/train.log 2>&1


# ps -ef | grep "train.sh"