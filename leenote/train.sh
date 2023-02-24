# deeptrain
# conda activate DeepPersonality
# cd /home/zl525/code/DeepPersonality/ && sh leenote/train.sh

cd /home/zl525/code/DeepPersonality/

#### **************************** transformer: transformer_udiva.yaml **************************** ####
# sample_size=16
# epoch=2
# batch_size=32
# learning_rate=0.0001

# python3 -u ./script/run_exp.py --cfg_file ./config/demo/transformer_udiva.yaml \
# --sample_size $sample_size \
# --max_epoch $epoch \
# --bs $batch_size \
# --lr $learning_rate

#### **************************** resnet3D: resnet_3d_udiva_full.yaml **************************** ####
sample_size=4
epoch=2
batch_size=5
learning_rate=0.0001

python3 -u ./script/run_exp.py --cfg_file ./config/demo/resnet_3d_udiva_full.yaml \
--sample_size $sample_size \
--max_epoch $epoch \
--bs $batch_size \
--lr $learning_rate

#### **************************** model2: bimodal_lstm_udiva_full.yaml **************************** ####
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
# --resume $resume

#### **************************** model1: bimodal_resnet18_udiva_full.yaml **************************** ####
# sample_size=20
# epoch=1
# batch_size=32
# learning_rate=0.0005

# python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
# --sample_size $sample_size \
# --max_epoch $epoch \
# --bs $batch_size \
# --lr $learning_rate \

