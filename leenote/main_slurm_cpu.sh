# cd /home/zl525/code/DeepPersonality/leenote && sbatch slurm_submit_deep.peta4-skylake

cd /home/zl525/code/DeepPersonality/


#### ===================== new: bimodal_lstm_udiva_full.yaml ===================== ####
sample_size=10
epoch=500
batch_size=1
learning_rate=0.01
cfg_file=./config/demo/bimodal_lstm_udiva_full.yaml

python3 ./script/run_exp.py --cfg_file $cfg_file \
--sample_size $sample_size \
--max_epoch $epoch \
--bs $batch_size \
--lr $learning_rate


### ===================== old: bimodal_resnet18_udiva_full.yaml ===================== ####
# sample_size=400
# epoch=50
# batch_size=16
# learning_rate=0.0001
# resume="results/demo/unified_frame_images/bimodal_resnet_udiva/02-03_21-46/checkpoint_0.pkl"

# python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
# --sample_size $sample_size \
# --max_epoch $epoch \
# --bs $batch_size \
# --lr $learning_rate \
# --resume $resume
