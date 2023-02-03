cd /home/zl525/code/DeepPersonality/


##==============================================================================##
# 测试每秒抽5帧时的结果
python3 ./script/run_exp.py \
--cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
--max_epoch 10 --bs 8

python3 ./script/run_exp.py \
--cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
--max_epoch 50 --bs 8

python3 ./script/run_exp.py \
--cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
--max_epoch 100 --bs 8

python3 ./script/run_exp.py \
--cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
--max_epoch 200 --bs 8

##==============================================================================##
# python3 ./script/run_exp.py \
# --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
# --max_epoch 10 --bs 2

# python3 ./script/run_exp.py \
# --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
# --max_epoch 10 --bs 4

# python3 ./script/run_exp.py \
# --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
# --max_epoch 10 --bs 6

# python3 ./script/run_exp.py \
# --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
# --max_epoch 10 --bs 8

# python3 ./script/run_exp.py \
# --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
# --max_epoch 10 --bs 10

# python3 ./script/run_exp.py \
# --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
# --max_epoch 10 --bs 12

# python3 ./script/run_exp.py \
# --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
# --max_epoch 10 --bs 14



##==============================================================================##
# python3 ./script/run_exp.py \
# --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
# --max_epoch 100 --bs 32

# python3 ./script/run_exp.py \
# --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
# --max_epoch 200 --bs 32

# python3 ./script/run_exp.py \
# --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml \
# --max_epoch 300 --bs 32