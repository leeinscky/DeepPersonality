# cd /home/zl525/code/DeepPersonality/leenote && sbatch slurm_submit_deep

cd /home/zl525/code/DeepPersonality/


#### ===================== resnet3D: resnet_3d_udiva_full.yaml ===================== ####
sample_size=80
epoch=200
batch_size=16
learning_rate=0.0005
cfg_file=./config/demo/resnet_3d_udiva_full.yaml

python3 ./script/run_exp.py --cfg_file $cfg_file \
--sample_size $sample_size \
--max_epoch $epoch \
--bs $batch_size \
--lr $learning_rate

#### ===================== new: bimodal_lstm_udiva_full.yaml ===================== ####
# sample_size=250
# epoch=50
# batch_size=16
# learning_rate=0.0001
# cfg_file=./config/demo/bimodal_lstm_udiva_full.yaml

# python3 ./script/run_exp.py --cfg_file $cfg_file \
# --sample_size $sample_size \
# --max_epoch $epoch \
# --bs $batch_size \
# --lr $learning_rate
