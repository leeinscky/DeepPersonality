# deeptrain
# conda activate DeepPersonality
# cd /home/zl525/code/DeepPersonality/ && sh leenote/train.sh

cd /home/zl525/code/DeepPersonality/

sample_size=3
epoch=1
batch_size=2
learning_rate=0.001

python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_lstm_udiva_full.yaml \
--sample_size $sample_size \
--max_epoch $epoch \
--bs $batch_size \
--lr $learning_rate \

# ===============================================================================================
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
