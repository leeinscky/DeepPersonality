# cd /home/zl525/code/DeepPersonality/ && conda activate DeepPersonality
# python3 ./leenote/slurm_test/slurm_test.py 29 1

import os
import torch
import sys
import time

print('sys.argv:', sys.argv)

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"[GPU INFO] Device {i}: {device_name}")
    print('[GPU INFO] Current Allocated GPU ID:', torch.cuda.current_device())
else:
    print("[GPU INFO] No GPU available!!!!!!!!")

# 检查是否有可用的 GPU，如果有则使用sys.argv[2]作为cuda id
device = torch.device(f"cuda:{sys.argv[2]}" if torch.cuda.is_available() else "cpu")

# use CUDA_VISIBLE_DEVICES to set the GPU id
# device = torch.device(f"cuda:{os.environ['CUDA_VISIBLE_DEVICES']}" if torch.cuda.is_available() else "cpu")

print('device:', device)

# 定义输入数据和模型参数
input_data = torch.rand((3, 3)).to(device)
weight = torch.rand((3, 3)).to(device)

# 进行矩阵乘法运算
output = torch.mm(input_data, weight)

# 输出结果以及output所在的设备id
print('mm output:', output, ', output device id:', output.device)

# # print all available gpu device ids and the number of gpu devices
# print('number of GPU devices:', torch.cuda.device_count())
# print('torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0))
# print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES']) 


print('sys.argv:', sys.argv)
num = sys.argv[1]
# print hello and current time with YYYY-MM-DD HH:MM:SS format
print('hello:', num, ', current time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# sleep 1 minutes
import time
time.sleep(30)

