cd /home/zl525/code/DeepPersonality/

python3 ./script/run_exp.py \
--cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
--max_epoch 100 --bs 32

python3 ./script/run_exp.py \
--cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
--max_epoch 200 --bs 32

python3 ./script/run_exp.py \
--cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
--max_epoch 300 --bs 32