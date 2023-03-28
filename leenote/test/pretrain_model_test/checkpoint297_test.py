# 原文链接：https://blog.csdn.net/hugvgngj/article/details/111186170

import torch  # 命令行是逐行立即执行的
content = torch.load('deeppersonality_resnet_pretrain_checkpoint_297.pkl', map_location='cpu')
print(content.keys())   # dict_keys(['model_state_dict', 'optimizer_state_dict', 'epoch','best_acc'])

print(f'content[epoch]: {content["epoch"]}') # 297
print(f'content[best_acc]: {content["best_acc"]}') # 0.8943645358085632

# print(f'type(content[optimizer_state_dict]): {type(content["optimizer_state_dict"])}') #  <class 'dict'>
# print(f'content[optimizer_state_dict].keys(): {content["optimizer_state_dict"].keys()}') # dict_keys(['state', 'param_groups'])
# print(f'content[optimizer_state_dict][state]: {content["optimizer_state_dict"]["state"].keys()}') # dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121])
# print(f'content["optimizer_state_dict"]["state"][0]: {content["optimizer_state_dict"]["state"][0]}')
# print(f'content[optimizer_state_dict][param_groups]: {content["optimizer_state_dict"]["param_groups"]}')


# print(f'content["model_state_dict"].keys(): {content["model_state_dict"].keys()}')


for key in content["model_state_dict"].keys():
    print(f'key: {key}, \t \t \t \t{content["model_state_dict"][key].shape}')

# print result: print(f'key: {key}, \t \t \t \t{content["model_state_dict"][key].shape}')

'''
key: audio_branch.init_stage.conv1.weight, 	 	 	 	torch.Size([32, 1, 1, 49])
key: audio_branch.init_stage.bn1.weight, 	 	 	 	torch.Size([32])
key: audio_branch.init_stage.bn1.bias, 	 	 	 	torch.Size([32])
key: audio_branch.init_stage.bn1.running_mean, 	 	 	 	torch.Size([32])
key: audio_branch.init_stage.bn1.running_var, 	 	 	 	torch.Size([32])
key: audio_branch.init_stage.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer1.0.conv1.weight, 	 	 	 	torch.Size([32, 32, 1, 9])
key: audio_branch.layer1.0.bn1.weight, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.0.bn1.bias, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.0.bn1.running_mean, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.0.bn1.running_var, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.0.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer1.0.conv2.weight, 	 	 	 	torch.Size([32, 32, 1, 9])
key: audio_branch.layer1.0.bn2.weight, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.0.bn2.bias, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.0.bn2.running_mean, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.0.bn2.running_var, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.0.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer1.1.conv1.weight, 	 	 	 	torch.Size([32, 32, 1, 9])
key: audio_branch.layer1.1.bn1.weight, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.1.bn1.bias, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.1.bn1.running_mean, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.1.bn1.running_var, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.1.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer1.1.conv2.weight, 	 	 	 	torch.Size([32, 32, 1, 9])
key: audio_branch.layer1.1.bn2.weight, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.1.bn2.bias, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.1.bn2.running_mean, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.1.bn2.running_var, 	 	 	 	torch.Size([32])
key: audio_branch.layer1.1.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer2.0.conv1.weight, 	 	 	 	torch.Size([64, 32, 1, 9])
key: audio_branch.layer2.0.bn1.weight, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.0.bn1.bias, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.0.bn1.running_mean, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.0.bn1.running_var, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.0.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer2.0.conv2.weight, 	 	 	 	torch.Size([64, 64, 1, 9])
key: audio_branch.layer2.0.bn2.weight, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.0.bn2.bias, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.0.bn2.running_mean, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.0.bn2.running_var, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.0.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer2.0.downsample.0.weight, 	 	 	 	torch.Size([64, 32, 1, 1])
key: audio_branch.layer2.0.downsample.1.weight, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.0.downsample.1.bias, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.0.downsample.1.running_mean, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.0.downsample.1.running_var, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.0.downsample.1.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer2.1.conv1.weight, 	 	 	 	torch.Size([64, 64, 1, 9])
key: audio_branch.layer2.1.bn1.weight, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.1.bn1.bias, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.1.bn1.running_mean, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.1.bn1.running_var, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.1.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer2.1.conv2.weight, 	 	 	 	torch.Size([64, 64, 1, 9])
key: audio_branch.layer2.1.bn2.weight, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.1.bn2.bias, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.1.bn2.running_mean, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.1.bn2.running_var, 	 	 	 	torch.Size([64])
key: audio_branch.layer2.1.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer3.0.conv1.weight, 	 	 	 	torch.Size([128, 64, 1, 9])
key: audio_branch.layer3.0.bn1.weight, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.0.bn1.bias, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.0.bn1.running_mean, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.0.bn1.running_var, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.0.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer3.0.conv2.weight, 	 	 	 	torch.Size([128, 128, 1, 9])
key: audio_branch.layer3.0.bn2.weight, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.0.bn2.bias, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.0.bn2.running_mean, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.0.bn2.running_var, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.0.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer3.0.downsample.0.weight, 	 	 	 	torch.Size([128, 64, 1, 1])
key: audio_branch.layer3.0.downsample.1.weight, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.0.downsample.1.bias, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.0.downsample.1.running_mean, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.0.downsample.1.running_var, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.0.downsample.1.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer3.1.conv1.weight, 	 	 	 	torch.Size([128, 128, 1, 9])
key: audio_branch.layer3.1.bn1.weight, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.1.bn1.bias, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.1.bn1.running_mean, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.1.bn1.running_var, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.1.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer3.1.conv2.weight, 	 	 	 	torch.Size([128, 128, 1, 9])
key: audio_branch.layer3.1.bn2.weight, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.1.bn2.bias, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.1.bn2.running_mean, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.1.bn2.running_var, 	 	 	 	torch.Size([128])
key: audio_branch.layer3.1.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer4.0.conv1.weight, 	 	 	 	torch.Size([256, 128, 1, 9])
key: audio_branch.layer4.0.bn1.weight, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.0.bn1.bias, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.0.bn1.running_mean, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.0.bn1.running_var, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.0.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer4.0.conv2.weight, 	 	 	 	torch.Size([256, 256, 1, 9])
key: audio_branch.layer4.0.bn2.weight, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.0.bn2.bias, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.0.bn2.running_mean, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.0.bn2.running_var, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.0.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer4.0.downsample.0.weight, 	 	 	 	torch.Size([256, 128, 1, 1])
key: audio_branch.layer4.0.downsample.1.weight, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.0.downsample.1.bias, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.0.downsample.1.running_mean, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.0.downsample.1.running_var, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.0.downsample.1.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer4.1.conv1.weight, 	 	 	 	torch.Size([256, 256, 1, 9])
key: audio_branch.layer4.1.bn1.weight, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.1.bn1.bias, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.1.bn1.running_mean, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.1.bn1.running_var, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.1.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: audio_branch.layer4.1.conv2.weight, 	 	 	 	torch.Size([256, 256, 1, 9])
key: audio_branch.layer4.1.bn2.weight, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.1.bn2.bias, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.1.bn2.running_mean, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.1.bn2.running_var, 	 	 	 	torch.Size([256])
key: audio_branch.layer4.1.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.init_stage.conv1.weight, 	 	 	 	torch.Size([32, 3, 7, 7])
key: visual_branch.init_stage.bn1.weight, 	 	 	 	torch.Size([32])
key: visual_branch.init_stage.bn1.bias, 	 	 	 	torch.Size([32])
key: visual_branch.init_stage.bn1.running_mean, 	 	 	 	torch.Size([32])
key: visual_branch.init_stage.bn1.running_var, 	 	 	 	torch.Size([32])
key: visual_branch.init_stage.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer1.0.conv1.weight, 	 	 	 	torch.Size([32, 32, 3, 3])
key: visual_branch.layer1.0.bn1.weight, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.0.bn1.bias, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.0.bn1.running_mean, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.0.bn1.running_var, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.0.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer1.0.conv2.weight, 	 	 	 	torch.Size([32, 32, 3, 3])
key: visual_branch.layer1.0.bn2.weight, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.0.bn2.bias, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.0.bn2.running_mean, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.0.bn2.running_var, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.0.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer1.1.conv1.weight, 	 	 	 	torch.Size([32, 32, 3, 3])
key: visual_branch.layer1.1.bn1.weight, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.1.bn1.bias, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.1.bn1.running_mean, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.1.bn1.running_var, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.1.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer1.1.conv2.weight, 	 	 	 	torch.Size([32, 32, 3, 3])
key: visual_branch.layer1.1.bn2.weight, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.1.bn2.bias, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.1.bn2.running_mean, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.1.bn2.running_var, 	 	 	 	torch.Size([32])
key: visual_branch.layer1.1.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer2.0.conv1.weight, 	 	 	 	torch.Size([64, 32, 3, 3])
key: visual_branch.layer2.0.bn1.weight, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.0.bn1.bias, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.0.bn1.running_mean, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.0.bn1.running_var, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.0.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer2.0.conv2.weight, 	 	 	 	torch.Size([64, 64, 3, 3])
key: visual_branch.layer2.0.bn2.weight, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.0.bn2.bias, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.0.bn2.running_mean, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.0.bn2.running_var, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.0.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer2.0.downsample.0.weight, 	 	 	 	torch.Size([64, 32, 1, 1])
key: visual_branch.layer2.0.downsample.1.weight, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.0.downsample.1.bias, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.0.downsample.1.running_mean, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.0.downsample.1.running_var, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.0.downsample.1.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer2.1.conv1.weight, 	 	 	 	torch.Size([64, 64, 3, 3])
key: visual_branch.layer2.1.bn1.weight, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.1.bn1.bias, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.1.bn1.running_mean, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.1.bn1.running_var, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.1.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer2.1.conv2.weight, 	 	 	 	torch.Size([64, 64, 3, 3])
key: visual_branch.layer2.1.bn2.weight, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.1.bn2.bias, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.1.bn2.running_mean, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.1.bn2.running_var, 	 	 	 	torch.Size([64])
key: visual_branch.layer2.1.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer3.0.conv1.weight, 	 	 	 	torch.Size([128, 64, 3, 3])
key: visual_branch.layer3.0.bn1.weight, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.0.bn1.bias, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.0.bn1.running_mean, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.0.bn1.running_var, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.0.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer3.0.conv2.weight, 	 	 	 	torch.Size([128, 128, 3, 3])
key: visual_branch.layer3.0.bn2.weight, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.0.bn2.bias, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.0.bn2.running_mean, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.0.bn2.running_var, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.0.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer3.0.downsample.0.weight, 	 	 	 	torch.Size([128, 64, 1, 1])
key: visual_branch.layer3.0.downsample.1.weight, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.0.downsample.1.bias, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.0.downsample.1.running_mean, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.0.downsample.1.running_var, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.0.downsample.1.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer3.1.conv1.weight, 	 	 	 	torch.Size([128, 128, 3, 3])
key: visual_branch.layer3.1.bn1.weight, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.1.bn1.bias, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.1.bn1.running_mean, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.1.bn1.running_var, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.1.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer3.1.conv2.weight, 	 	 	 	torch.Size([128, 128, 3, 3])
key: visual_branch.layer3.1.bn2.weight, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.1.bn2.bias, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.1.bn2.running_mean, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.1.bn2.running_var, 	 	 	 	torch.Size([128])
key: visual_branch.layer3.1.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer4.0.conv1.weight, 	 	 	 	torch.Size([256, 128, 3, 3])
key: visual_branch.layer4.0.bn1.weight, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.0.bn1.bias, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.0.bn1.running_mean, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.0.bn1.running_var, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.0.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer4.0.conv2.weight, 	 	 	 	torch.Size([256, 256, 3, 3])
key: visual_branch.layer4.0.bn2.weight, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.0.bn2.bias, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.0.bn2.running_mean, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.0.bn2.running_var, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.0.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer4.0.downsample.0.weight, 	 	 	 	torch.Size([256, 128, 1, 1])
key: visual_branch.layer4.0.downsample.1.weight, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.0.downsample.1.bias, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.0.downsample.1.running_mean, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.0.downsample.1.running_var, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.0.downsample.1.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer4.1.conv1.weight, 	 	 	 	torch.Size([256, 256, 3, 3])
key: visual_branch.layer4.1.bn1.weight, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.1.bn1.bias, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.1.bn1.running_mean, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.1.bn1.running_var, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.1.bn1.num_batches_tracked, 	 	 	 	torch.Size([])
key: visual_branch.layer4.1.conv2.weight, 	 	 	 	torch.Size([256, 256, 3, 3])
key: visual_branch.layer4.1.bn2.weight, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.1.bn2.bias, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.1.bn2.running_mean, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.1.bn2.running_var, 	 	 	 	torch.Size([256])
key: visual_branch.layer4.1.bn2.num_batches_tracked, 	 	 	 	torch.Size([])
key: linear.weight, 	 	 	 	torch.Size([5, 512])
key: linear.bias, 	 	 	 	torch.Size([5])


# 作为对比，我构造的的模型用来处理udiva的visual_branch参数为
[dpcv/modeling/networks/audio_visual_residual.py] print_model_weights...
init_stage.conv1.weight torch.Size([32, 6, 7, 7])
init_stage.bn1.weight torch.Size([32])
init_stage.bn1.bias torch.Size([32])
layer1.0.conv1.weight torch.Size([32, 32, 3, 3])
layer1.0.bn1.weight torch.Size([32])
layer1.0.bn1.bias torch.Size([32])
layer1.0.conv2.weight torch.Size([32, 32, 3, 3])
layer1.0.bn2.weight torch.Size([32])
layer1.0.bn2.bias torch.Size([32])
layer1.1.conv1.weight torch.Size([32, 32, 3, 3])
layer1.1.bn1.weight torch.Size([32])
layer1.1.bn1.bias torch.Size([32])
layer1.1.conv2.weight torch.Size([32, 32, 3, 3])
layer1.1.bn2.weight torch.Size([32])
layer1.1.bn2.bias torch.Size([32])
layer2.0.conv1.weight torch.Size([64, 32, 3, 3])
layer2.0.bn1.weight torch.Size([64])
layer2.0.bn1.bias torch.Size([64])
layer2.0.conv2.weight torch.Size([64, 64, 3, 3])
layer2.0.bn2.weight torch.Size([64])
layer2.0.bn2.bias torch.Size([64])
layer2.0.downsample.0.weight torch.Size([64, 32, 1, 1])
layer2.0.downsample.1.weight torch.Size([64])
layer2.0.downsample.1.bias torch.Size([64])
layer2.1.conv1.weight torch.Size([64, 64, 3, 3])
layer2.1.bn1.weight torch.Size([64])
layer2.1.bn1.bias torch.Size([64])
layer2.1.conv2.weight torch.Size([64, 64, 3, 3])
layer2.1.bn2.weight torch.Size([64])
layer2.1.bn2.bias torch.Size([64])
layer3.0.conv1.weight torch.Size([128, 64, 3, 3])
layer3.0.bn1.weight torch.Size([128])
layer3.0.bn1.bias torch.Size([128])
layer3.0.conv2.weight torch.Size([128, 128, 3, 3])
layer3.0.bn2.weight torch.Size([128])
layer3.0.bn2.bias torch.Size([128])
layer3.0.downsample.0.weight torch.Size([128, 64, 1, 1])
layer3.0.downsample.1.weight torch.Size([128])
layer3.0.downsample.1.bias torch.Size([128])
layer3.1.conv1.weight torch.Size([128, 128, 3, 3])
layer3.1.bn1.weight torch.Size([128])
layer3.1.bn1.bias torch.Size([128])
layer3.1.conv2.weight torch.Size([128, 128, 3, 3])
layer3.1.bn2.weight torch.Size([128])
layer3.1.bn2.bias torch.Size([128])
layer4.0.conv1.weight torch.Size([256, 128, 3, 3])
layer4.0.bn1.weight torch.Size([256])
layer4.0.bn1.bias torch.Size([256])
layer4.0.conv2.weight torch.Size([256, 256, 3, 3])
layer4.0.bn2.weight torch.Size([256])
layer4.0.bn2.bias torch.Size([256])
layer4.0.downsample.0.weight torch.Size([256, 128, 1, 1])
layer4.0.downsample.1.weight torch.Size([256])
layer4.0.downsample.1.bias torch.Size([256])
layer4.1.conv1.weight torch.Size([256, 256, 3, 3])
layer4.1.bn1.weight torch.Size([256])
layer4.1.bn1.bias torch.Size([256])
layer4.1.conv2.weight torch.Size([256, 256, 3, 3])
layer4.1.bn2.weight torch.Size([256])
layer4.1.bn2.bias torch.Size([256])
lstm.weight_ih_l0 torch.Size([1024, 256])
lstm.weight_hh_l0 torch.Size([1024, 256])
lstm.bias_ih_l0 torch.Size([1024])
lstm.bias_hh_l0 torch.Size([1024])
lstm.weight_ih_l1 torch.Size([1024, 256])
lstm.weight_hh_l1 torch.Size([1024, 256])
lstm.bias_ih_l1 torch.Size([1024])
lstm.bias_hh_l1 torch.Size([1024])
lstm.weight_ih_l2 torch.Size([1024, 256])
lstm.weight_hh_l2 torch.Size([1024, 256])
lstm.bias_ih_l2 torch.Size([1024])
lstm.bias_hh_l2 torch.Size([1024])
fc1.weight torch.Size([128, 256])
fc1.bias torch.Size([128])
fc2.weight torch.Size([1, 128])
fc2.bias torch.Size([1])

# 在 dpcv/experiment/exp_runner.py 中加载该checkpoint，会报错如下：
RuntimeError: Error(s) in loading state_dict for AudioVisualResNet18LSTMUdiva:
	Missing key(s) in state_dict: "audio_branch.lstm.weight_ih_l0", 
     "audio_branch.lstm.weight_hh_l0", 
     "audio_branch.lstm.bias_ih_l0", 
     "audio_branch.lstm.bias_hh_l0", 
     "audio_branch.lstm.weight_ih_l1", 
     "audio_branch.lstm.weight_hh_l1", 
     "audio_branch.lstm.bias_ih_l1", 
     "audio_branch.lstm.bias_hh_l1", 
     "audio_branch.lstm.weight_ih_l2", 
     "audio_branch.lstm.weight_hh_l2", 
     "audio_branch.lstm.bias_ih_l2", 
     "audio_branch.lstm.bias_hh_l2", 
     "audio_branch.fc1.weight", 
     "audio_branch.fc1.bias", 
     "audio_branch.fc2.weight", 
     "audio_branch.fc2.bias", 
     "visual_branch.lstm.weight_ih_l0", 
     "visual_branch.lstm.weight_hh_l0", 
     "visual_branch.lstm.bias_ih_l0", 
     "visual_branch.lstm.bias_hh_l0", 
     "visual_branch.lstm.weight_ih_l1", 
     "visual_branch.lstm.weight_hh_l1", 
     "visual_branch.lstm.bias_ih_l1", 
     "visual_branch.lstm.bias_hh_l1", 
     "visual_branch.lstm.weight_ih_l2", 
     "visual_branch.lstm.weight_hh_l2", 
     "visual_branch.lstm.bias_ih_l2", 
     "visual_branch.lstm.bias_hh_l2", 
     "visual_branch.fc1.weight", 
     "visual_branch.fc1.bias", 
     "visual_branch.fc2.weight", 
     "visual_branch.fc2.bias", 
     "bn.weight", 
     "bn.bias", 
     "bn.running_mean", 
     "bn.running_var".
	size mismatch for audio_branch.init_stage.conv1.weight: copying a param with shape torch.Size([32, 1, 1, 49]) from checkpoint, the shape in current model is torch.Size([32, 2, 1, 49]).
	size mismatch for visual_branch.init_stage.conv1.weight: copying a param with shape torch.Size([32, 3, 7, 7]) from checkpoint, the shape in current model is torch.Size([32, 6, 7, 7]).
	size mismatch for linear.weight: copying a param with shape torch.Size([5, 512]) from checkpoint, the shape in current model is torch.Size([2, 256]).
	size mismatch for linear.bias: copying a param with shape torch.Size([5]) from checkpoint, the shape in current model is torch.Size([2]).

# 如果在dpcv/modeling/networks/audio_visual_residual.py中正常加载该checkpoint，会报错如下：
RuntimeError: Error(s) in loading state_dict for AudioVisualResNetUdiva:
	Missing key(s) in state_dict: 
     "init_stage.conv1.weight", 
     "init_stage.bn1.weight", 
     "init_stage.bn1.bias", 
     "init_stage.bn1.running_mean", 
     "init_stage.bn1.running_var", 
     "layer1.0.conv1.weight", 
     "layer1.0.bn1.weight", 
     "layer1.0.bn1.bias", 
     "layer1.0.bn1.running_mean", 
     "layer1.0.bn1.running_var", 
     "layer1.0.conv2.weight", 
     "layer1.0.bn2.weight", 
     "layer1.0.bn2.bias", 
     "layer1.0.bn2.running_mean", 
     "layer1.0.bn2.running_var", 
     "layer1.1.conv1.weight", 
     "layer1.1.bn1.weight", 
     "layer1.1.bn1.bias", 
     "layer1.1.bn1.running_mean", 
     "layer1.1.bn1.running_var", 
     "layer1.1.conv2.weight", 
     "layer1.1.bn2.weight", 
     "layer1.1.bn2.bias", 
     "layer1.1.bn2.running_mean", 
     "layer1.1.bn2.running_var", 
     "layer2.0.conv1.weight", 
     "layer2.0.bn1.weight", 
     "layer2.0.bn1.bias", 
     "layer2.0.bn1.running_mean", 
     "layer2.0.bn1.running_var", 
     "layer2.0.conv2.weight", 
     "layer2.0.bn2.weight", 
     "layer2.0.bn2.bias", 
     "layer2.0.bn2.running_mean", 
     "layer2.0.bn2.running_var", 
     "layer2.0.downsample.0.weight", 
     "layer2.0.downsample.1.weight", 
     "layer2.0.downsample.1.bias", 
     "layer2.0.downsample.1.running_mean", 
     "layer2.0.downsample.1.running_var", 
     "layer2.1.conv1.weight", 
     "layer2.1.bn1.weight", 
     "layer2.1.bn1.bias", 
     "layer2.1.bn1.running_mean", 
     "layer2.1.bn1.running_var", 
     "layer2.1.conv2.weight", 
     "layer2.1.bn2.weight", 
     "layer2.1.bn2.bias", 
     "layer2.1.bn2.running_mean", 
     "layer2.1.bn2.running_var", 
     "layer3.0.conv1.weight", 
     "layer3.0.bn1.weight", 
     "layer3.0.bn1.bias", 
     "layer3.0.bn1.running_mean", 
     "layer3.0.bn1.running_var", 
     "layer3.0.conv2.weight", 
     "layer3.0.bn2.weight", 
     "layer3.0.bn2.bias", 
     "layer3.0.bn2.running_mean", 
     "layer3.0.bn2.running_var", 
     "layer3.0.downsample.0.weight", 
     "layer3.0.downsample.1.weight", 
     "layer3.0.downsample.1.bias", 
     "layer3.0.downsample.1.running_mean", 
     "layer3.0.downsample.1.running_var", 
     "layer3.1.conv1.weight", 
     "layer3.1.bn1.weight", 
     "layer3.1.bn1.bias", 
     "layer3.1.bn1.running_mean", 
     "layer3.1.bn1.running_var", 
     "layer3.1.conv2.weight", 
     "layer3.1.bn2.weight", 
     "layer3.1.bn2.bias", 
     "layer3.1.bn2.running_mean", 
     "layer3.1.bn2.running_var", 
     "layer4.0.conv1.weight", 
     "layer4.0.bn1.weight", 
     "layer4.0.bn1.bias", 
     "layer4.0.bn1.running_mean", 
     "layer4.0.bn1.running_var", 
     "layer4.0.conv2.weight", 
     "layer4.0.bn2.weight", 
     "layer4.0.bn2.bias", 
     "layer4.0.bn2.running_mean", 
     "layer4.0.bn2.running_var", 
     "layer4.0.downsample.0.weight", 
     "layer4.0.downsample.1.weight", 
     "layer4.0.downsample.1.bias", 
     "layer4.0.downsample.1.running_mean", 
     "layer4.0.downsample.1.running_var", 
     "layer4.1.conv1.weight", 
     "layer4.1.bn1.weight", 
     "layer4.1.bn1.bias", 
     "layer4.1.bn1.running_mean", 
     "layer4.1.bn1.running_var", 
     "layer4.1.conv2.weight", 
     "layer4.1.bn2.weight", 
     "layer4.1.bn2.bias", 
     "layer4.1.bn2.running_mean", 
     "layer4.1.bn2.running_var", 
     "lstm.weight_ih_l0", 
     "lstm.weight_hh_l0", 
     "lstm.bias_ih_l0", 
     "lstm.bias_hh_l0", 
     "lstm.weight_ih_l1", 
     "lstm.weight_hh_l1", 
     "lstm.bias_ih_l1", 
     "lstm.bias_hh_l1", 
     "lstm.weight_ih_l2", 
     "lstm.weight_hh_l2", 
     "lstm.bias_ih_l2", 
     "lstm.bias_hh_l2", 
     "fc1.weight", 
     "fc1.bias", 
     "fc2.weight", 
     "fc2.bias".
	
 
 
 Unexpected key(s) in state_dict: "audio_branch.init_stage.conv1.weight", 
     "audio_branch.init_stage.bn1.weight", 
     "audio_branch.init_stage.bn1.bias", 
     "audio_branch.init_stage.bn1.running_mean", 
     "audio_branch.init_stage.bn1.running_var", 
     "audio_branch.init_stage.bn1.num_batches_tracked", 
     "audio_branch.layer1.0.conv1.weight", 
     "audio_branch.layer1.0.bn1.weight", 
     "audio_branch.layer1.0.bn1.bias", 
     "audio_branch.layer1.0.bn1.running_mean", 
     "audio_branch.layer1.0.bn1.running_var", 
     "audio_branch.layer1.0.bn1.num_batches_tracked", 
     "audio_branch.layer1.0.conv2.weight", 
     "audio_branch.layer1.0.bn2.weight", 
     "audio_branch.layer1.0.bn2.bias", 
     "audio_branch.layer1.0.bn2.running_mean", 
     "audio_branch.layer1.0.bn2.running_var", 
     "audio_branch.layer1.0.bn2.num_batches_tracked", 
     "audio_branch.layer1.1.conv1.weight", 
     "audio_branch.layer1.1.bn1.weight", 
     "audio_branch.layer1.1.bn1.bias", 
     "audio_branch.layer1.1.bn1.running_mean", 
     "audio_branch.layer1.1.bn1.running_var", 
     "audio_branch.layer1.1.bn1.num_batches_tracked", 
     "audio_branch.layer1.1.conv2.weight", 
     "audio_branch.layer1.1.bn2.weight", 
     "audio_branch.layer1.1.bn2.bias", 
     "audio_branch.layer1.1.bn2.running_mean", 
     "audio_branch.layer1.1.bn2.running_var", 
     "audio_branch.layer1.1.bn2.num_batches_tracked", 
     "audio_branch.layer2.0.conv1.weight", 
     "audio_branch.layer2.0.bn1.weight", 
     "audio_branch.layer2.0.bn1.bias", 
     "audio_branch.layer2.0.bn1.running_mean", 
     "audio_branch.layer2.0.bn1.running_var", 
     "audio_branch.layer2.0.bn1.num_batches_tracked", 
     "audio_branch.layer2.0.conv2.weight", 
     "audio_branch.layer2.0.bn2.weight", 
     "audio_branch.layer2.0.bn2.bias", 
     "audio_branch.layer2.0.bn2.running_mean", 
     "audio_branch.layer2.0.bn2.running_var", 
     "audio_branch.layer2.0.bn2.num_batches_tracked", 
     "audio_branch.layer2.0.downsample.0.weight", 
     "audio_branch.layer2.0.downsample.1.weight", 
     "audio_branch.layer2.0.downsample.1.bias", 
     "audio_branch.layer2.0.downsample.1.running_mean", 
     "audio_branch.layer2.0.downsample.1.running_var", 
     "audio_branch.layer2.0.downsample.1.num_batches_tracked", 
     "audio_branch.layer2.1.conv1.weight", 
     "audio_branch.layer2.1.bn1.weight", 
     "audio_branch.layer2.1.bn1.bias", 
     "audio_branch.layer2.1.bn1.running_mean", 
     "audio_branch.layer2.1.bn1.running_var", 
     "audio_branch.layer2.1.bn1.num_batches_tracked", 
     "audio_branch.layer2.1.conv2.weight", 
     "audio_branch.layer2.1.bn2.weight", 
     "audio_branch.layer2.1.bn2.bias", 
     "audio_branch.layer2.1.bn2.running_mean", 
     "audio_branch.layer2.1.bn2.running_var", 
     "audio_branch.layer2.1.bn2.num_batches_tracked", 
     "audio_branch.layer3.0.conv1.weight", 
     "audio_branch.layer3.0.bn1.weight", 
     "audio_branch.layer3.0.bn1.bias", 
     "audio_branch.layer3.0.bn1.running_mean", 
     "audio_branch.layer3.0.bn1.running_var", 
     "audio_branch.layer3.0.bn1.num_batches_tracked", 
     "audio_branch.layer3.0.conv2.weight", 
     "audio_branch.layer3.0.bn2.weight", 
     "audio_branch.layer3.0.bn2.bias", 
     "audio_branch.layer3.0.bn2.running_mean", 
     "audio_branch.layer3.0.bn2.running_var", 
     "audio_branch.layer3.0.bn2.num_batches_tracked", 
     "audio_branch.layer3.0.downsample.0.weight", 
     "audio_branch.layer3.0.downsample.1.weight", 
     "audio_branch.layer3.0.downsample.1.bias", 
     "audio_branch.layer3.0.downsample.1.running_mean", 
     "audio_branch.layer3.0.downsample.1.running_var", 
     "audio_branch.layer3.0.downsample.1.num_batches_tracked", 
     "audio_branch.layer3.1.conv1.weight", 
     "audio_branch.layer3.1.bn1.weight", 
     "audio_branch.layer3.1.bn1.bias", 
     "audio_branch.layer3.1.bn1.running_mean", 
     "audio_branch.layer3.1.bn1.running_var", 
     "audio_branch.layer3.1.bn1.num_batches_tracked", 
     "audio_branch.layer3.1.conv2.weight", 
     "audio_branch.layer3.1.bn2.weight", 
     "audio_branch.layer3.1.bn2.bias", 
     "audio_branch.layer3.1.bn2.running_mean", 
     "audio_branch.layer3.1.bn2.running_var", 
     "audio_branch.layer3.1.bn2.num_batches_tracked", 
     "audio_branch.layer4.0.conv1.weight", 
     "audio_branch.layer4.0.bn1.weight", 
     "audio_branch.layer4.0.bn1.bias", 
     "audio_branch.layer4.0.bn1.running_mean", 
     "audio_branch.layer4.0.bn1.running_var", 
     "audio_branch.layer4.0.bn1.num_batches_tracked", 
     "audio_branch.layer4.0.conv2.weight", 
     "audio_branch.layer4.0.bn2.weight", 
     "audio_branch.layer4.0.bn2.bias", 
     "audio_branch.layer4.0.bn2.running_mean", 
     "audio_branch.layer4.0.bn2.running_var", 
     "audio_branch.layer4.0.bn2.num_batches_tracked", 
     "audio_branch.layer4.0.downsample.0.weight", 
     "audio_branch.layer4.0.downsample.1.weight", 
     "audio_branch.layer4.0.downsample.1.bias", 
     "audio_branch.layer4.0.downsample.1.running_mean", 
     "audio_branch.layer4.0.downsample.1.running_var", 
     "audio_branch.layer4.0.downsample.1.num_batches_tracked", 
     "audio_branch.layer4.1.conv1.weight", 
     "audio_branch.layer4.1.bn1.weight", 
     "audio_branch.layer4.1.bn1.bias", 
     "audio_branch.layer4.1.bn1.running_mean", 
     "audio_branch.layer4.1.bn1.running_var", 
     "audio_branch.layer4.1.bn1.num_batches_tracked", 
     "audio_branch.layer4.1.conv2.weight", 
     "audio_branch.layer4.1.bn2.weight", 
     "audio_branch.layer4.1.bn2.bias", 
     "audio_branch.layer4.1.bn2.running_mean", 
     "audio_branch.layer4.1.bn2.running_var", 
     "audio_branch.layer4.1.bn2.num_batches_tracked", 
     "visual_branch.init_stage.conv1.weight", 
     "visual_branch.init_stage.bn1.weight", 
     "visual_branch.init_stage.bn1.bias", 
     "visual_branch.init_stage.bn1.running_mean", 
     "visual_branch.init_stage.bn1.running_var", 
     "visual_branch.init_stage.bn1.num_batches_tracked", 
     "visual_branch.layer1.0.conv1.weight", 
     "visual_branch.layer1.0.bn1.weight", 
     "visual_branch.layer1.0.bn1.bias", 
     "visual_branch.layer1.0.bn1.running_mean", 
     "visual_branch.layer1.0.bn1.running_var", 
     "visual_branch.layer1.0.bn1.num_batches_tracked", 
     "visual_branch.layer1.0.conv2.weight", 
     "visual_branch.layer1.0.bn2.weight", 
     "visual_branch.layer1.0.bn2.bias", 
     "visual_branch.layer1.0.bn2.running_mean", 
     "visual_branch.layer1.0.bn2.running_var", 
     "visual_branch.layer1.0.bn2.num_batches_tracked", 
     "visual_branch.layer1.1.conv1.weight", 
     "visual_branch.layer1.1.bn1.weight", 
     "visual_branch.layer1.1.bn1.bias", 
     "visual_branch.layer1.1.bn1.running_mean", 
     "visual_branch.layer1.1.bn1.running_var", 
     "visual_branch.layer1.1.bn1.num_batches_tracked", 
     "visual_branch.layer1.1.conv2.weight", 
     "visual_branch.layer1.1.bn2.weight", 
     "visual_branch.layer1.1.bn2.bias", 
     "visual_branch.layer1.1.bn2.running_mean", 
     "visual_branch.layer1.1.bn2.running_var", 
     "visual_branch.layer1.1.bn2.num_batches_tracked", 
     "visual_branch.layer2.0.conv1.weight", 
     "visual_branch.layer2.0.bn1.weight", 
     "visual_branch.layer2.0.bn1.bias", 
     "visual_branch.layer2.0.bn1.running_mean", 
     "visual_branch.layer2.0.bn1.running_var", 
     "visual_branch.layer2.0.bn1.num_batches_tracked", 
     "visual_branch.layer2.0.conv2.weight", 
     "visual_branch.layer2.0.bn2.weight", 
     "visual_branch.layer2.0.bn2.bias", 
     "visual_branch.layer2.0.bn2.running_mean", 
     "visual_branch.layer2.0.bn2.running_var", 
     "visual_branch.layer2.0.bn2.num_batches_tracked", 
     "visual_branch.layer2.0.downsample.0.weight", 
     "visual_branch.layer2.0.downsample.1.weight", 
     "visual_branch.layer2.0.downsample.1.bias", 
     "visual_branch.layer2.0.downsample.1.running_mean", 
     "visual_branch.layer2.0.downsample.1.running_var", 
     "visual_branch.layer2.0.downsample.1.num_batches_tracked", 
     "visual_branch.layer2.1.conv1.weight", 
     "visual_branch.layer2.1.bn1.weight", 
     "visual_branch.layer2.1.bn1.bias", 
     "visual_branch.layer2.1.bn1.running_mean", 
     "visual_branch.layer2.1.bn1.running_var", 
     "visual_branch.layer2.1.bn1.num_batches_tracked", 
     "visual_branch.layer2.1.conv2.weight", 
     "visual_branch.layer2.1.bn2.weight", 
     "visual_branch.layer2.1.bn2.bias", 
     "visual_branch.layer2.1.bn2.running_mean", 
     "visual_branch.layer2.1.bn2.running_var", 
     "visual_branch.layer2.1.bn2.num_batches_tracked", 
     "visual_branch.layer3.0.conv1.weight", 
     "visual_branch.layer3.0.bn1.weight", 
     "visual_branch.layer3.0.bn1.bias", 
     "visual_branch.layer3.0.bn1.running_mean", 
     "visual_branch.layer3.0.bn1.running_var", 
     "visual_branch.layer3.0.bn1.num_batches_tracked", 
     "visual_branch.layer3.0.conv2.weight", 
     "visual_branch.layer3.0.bn2.weight", 
     "visual_branch.layer3.0.bn2.bias", 
     "visual_branch.layer3.0.bn2.running_mean", 
     "visual_branch.layer3.0.bn2.running_var", 
     "visual_branch.layer3.0.bn2.num_batches_tracked", 
     "visual_branch.layer3.0.downsample.0.weight", 
     "visual_branch.layer3.0.downsample.1.weight", 
     "visual_branch.layer3.0.downsample.1.bias", 
     "visual_branch.layer3.0.downsample.1.running_mean", 
     "visual_branch.layer3.0.downsample.1.running_var", 
     "visual_branch.layer3.0.downsample.1.num_batches_tracked", 
     "visual_branch.layer3.1.conv1.weight", 
     "visual_branch.layer3.1.bn1.weight", 
     "visual_branch.layer3.1.bn1.bias", 
     "visual_branch.layer3.1.bn1.running_mean", 
     "visual_branch.layer3.1.bn1.running_var", 
     "visual_branch.layer3.1.bn1.num_batches_tracked", 
     "visual_branch.layer3.1.conv2.weight", 
     "visual_branch.layer3.1.bn2.weight", 
     "visual_branch.layer3.1.bn2.bias", 
     "visual_branch.layer3.1.bn2.running_mean", 
     "visual_branch.layer3.1.bn2.running_var", 
     "visual_branch.layer3.1.bn2.num_batches_tracked", 
     "visual_branch.layer4.0.conv1.weight", 
     "visual_branch.layer4.0.bn1.weight", 
     "visual_branch.layer4.0.bn1.bias", 
     "visual_branch.layer4.0.bn1.running_mean", 
     "visual_branch.layer4.0.bn1.running_var", 
     "visual_branch.layer4.0.bn1.num_batches_tracked", 
     "visual_branch.layer4.0.conv2.weight", 
     "visual_branch.layer4.0.bn2.weight", 
     "visual_branch.layer4.0.bn2.bias", 
     "visual_branch.layer4.0.bn2.running_mean", 
     "visual_branch.layer4.0.bn2.running_var", 
     "visual_branch.layer4.0.bn2.num_batches_tracked", 
     "visual_branch.layer4.0.downsample.0.weight", 
     "visual_branch.layer4.0.downsample.1.weight", 
     "visual_branch.layer4.0.downsample.1.bias", 
     "visual_branch.layer4.0.downsample.1.running_mean", 
     "visual_branch.layer4.0.downsample.1.running_var", 
     "visual_branch.layer4.0.downsample.1.num_batches_tracked", 
     "visual_branch.layer4.1.conv1.weight", 
     "visual_branch.layer4.1.bn1.weight", 
     "visual_branch.layer4.1.bn1.bias", 
     "visual_branch.layer4.1.bn1.running_mean", 
     "visual_branch.layer4.1.bn1.running_var", 
     "visual_branch.layer4.1.bn1.num_batches_tracked", 
     "visual_branch.layer4.1.conv2.weight", 
     "visual_branch.layer4.1.bn2.weight", 
     "visual_branch.layer4.1.bn2.bias", 
     "visual_branch.layer4.1.bn2.running_mean", 
     "visual_branch.layer4.1.bn2.running_var", 
     "visual_branch.layer4.1.bn2.num_batches_tracked", 
     "linear.weight", 
     "linear.bias".
'''



# print result: print(f'content["model_state_dict"].keys(): {content["model_state_dict"].keys()}')
'''
content["model_state_dict"].keys(): odict_keys(
    ['audio_branch.init_stage.conv1.weight',
     'audio_branch.init_stage.bn1.weight',
     'audio_branch.init_stage.bn1.bias',
     'audio_branch.init_stage.bn1.running_mean',
     'audio_branch.init_stage.bn1.running_var',
     'audio_branch.init_stage.bn1.num_batches_tracked',
     'audio_branch.layer1.0.conv1.weight',
     'audio_branch.layer1.0.bn1.weight',
     'audio_branch.layer1.0.bn1.bias',
     'audio_branch.layer1.0.bn1.running_mean',
     'audio_branch.layer1.0.bn1.running_var',
     'audio_branch.layer1.0.bn1.num_batches_tracked',
     'audio_branch.layer1.0.conv2.weight',
     'audio_branch.layer1.0.bn2.weight',
     'audio_branch.layer1.0.bn2.bias',
     'audio_branch.layer1.0.bn2.running_mean',
     'audio_branch.layer1.0.bn2.running_var',
     'audio_branch.layer1.0.bn2.num_batches_tracked',
     'audio_branch.layer1.1.conv1.weight',
     'audio_branch.layer1.1.bn1.weight',
     'audio_branch.layer1.1.bn1.bias',
     'audio_branch.layer1.1.bn1.running_mean',
     'audio_branch.layer1.1.bn1.running_var',
     'audio_branch.layer1.1.bn1.num_batches_tracked',
     'audio_branch.layer1.1.conv2.weight',
     'audio_branch.layer1.1.bn2.weight',
     'audio_branch.layer1.1.bn2.bias',
     'audio_branch.layer1.1.bn2.running_mean',
     'audio_branch.layer1.1.bn2.running_var',
     'audio_branch.layer1.1.bn2.num_batches_tracked',
     'audio_branch.layer2.0.conv1.weight',
     'audio_branch.layer2.0.bn1.weight',
     'audio_branch.layer2.0.bn1.bias',
     'audio_branch.layer2.0.bn1.running_mean',
     'audio_branch.layer2.0.bn1.running_var',
     'audio_branch.layer2.0.bn1.num_batches_tracked',
     'audio_branch.layer2.0.conv2.weight',
     'audio_branch.layer2.0.bn2.weight',
     'audio_branch.layer2.0.bn2.bias',
     'audio_branch.layer2.0.bn2.running_mean',
     'audio_branch.layer2.0.bn2.running_var',
     'audio_branch.layer2.0.bn2.num_batches_tracked',
     'audio_branch.layer2.0.downsample.0.weight',
     'audio_branch.layer2.0.downsample.1.weight',
     'audio_branch.layer2.0.downsample.1.bias',
     'audio_branch.layer2.0.downsample.1.running_mean',
     'audio_branch.layer2.0.downsample.1.running_var',
     'audio_branch.layer2.0.downsample.1.num_batches_tracked',
     'audio_branch.layer2.1.conv1.weight',
     'audio_branch.layer2.1.bn1.weight',
     'audio_branch.layer2.1.bn1.bias',
     'audio_branch.layer2.1.bn1.running_mean',
     'audio_branch.layer2.1.bn1.running_var',
     'audio_branch.layer2.1.bn1.num_batches_tracked',
     'audio_branch.layer2.1.conv2.weight',
     'audio_branch.layer2.1.bn2.weight',
     'audio_branch.layer2.1.bn2.bias',
     'audio_branch.layer2.1.bn2.running_mean',
     'audio_branch.layer2.1.bn2.running_var',
     'audio_branch.layer2.1.bn2.num_batches_tracked',
     'audio_branch.layer3.0.conv1.weight',
     'audio_branch.layer3.0.bn1.weight',
     'audio_branch.layer3.0.bn1.bias',
     'audio_branch.layer3.0.bn1.running_mean',
     'audio_branch.layer3.0.bn1.running_var',
     'audio_branch.layer3.0.bn1.num_batches_tracked',
     'audio_branch.layer3.0.conv2.weight',
     'audio_branch.layer3.0.bn2.weight',
     'audio_branch.layer3.0.bn2.bias',
     'audio_branch.layer3.0.bn2.running_mean',
     'audio_branch.layer3.0.bn2.running_var',
     'audio_branch.layer3.0.bn2.num_batches_tracked',
     'audio_branch.layer3.0.downsample.0.weight',
     'audio_branch.layer3.0.downsample.1.weight',
     'audio_branch.layer3.0.downsample.1.bias',
     'audio_branch.layer3.0.downsample.1.running_mean',
     'audio_branch.layer3.0.downsample.1.running_var',
     'audio_branch.layer3.0.downsample.1.num_batches_tracked',
     'audio_branch.layer3.1.conv1.weight',
     'audio_branch.layer3.1.bn1.weight',
     'audio_branch.layer3.1.bn1.bias',
     'audio_branch.layer3.1.bn1.running_mean',
     'audio_branch.layer3.1.bn1.running_var',
     'audio_branch.layer3.1.bn1.num_batches_tracked',
     'audio_branch.layer3.1.conv2.weight',
     'audio_branch.layer3.1.bn2.weight',
     'audio_branch.layer3.1.bn2.bias',
     'audio_branch.layer3.1.bn2.running_mean',
     'audio_branch.layer3.1.bn2.running_var',
     'audio_branch.layer3.1.bn2.num_batches_tracked',
     'audio_branch.layer4.0.conv1.weight',
     'audio_branch.layer4.0.bn1.weight',
     'audio_branch.layer4.0.bn1.bias',
     'audio_branch.layer4.0.bn1.running_mean',
     'audio_branch.layer4.0.bn1.running_var',
     'audio_branch.layer4.0.bn1.num_batches_tracked',
     'audio_branch.layer4.0.conv2.weight',
     'audio_branch.layer4.0.bn2.weight',
     'audio_branch.layer4.0.bn2.bias',
     'audio_branch.layer4.0.bn2.running_mean',
     'audio_branch.layer4.0.bn2.running_var',
     'audio_branch.layer4.0.bn2.num_batches_tracked',
     'audio_branch.layer4.0.downsample.0.weight',
     'audio_branch.layer4.0.downsample.1.weight',
     'audio_branch.layer4.0.downsample.1.bias',
     'audio_branch.layer4.0.downsample.1.running_mean',
     'audio_branch.layer4.0.downsample.1.running_var',
     'audio_branch.layer4.0.downsample.1.num_batches_tracked',
     'audio_branch.layer4.1.conv1.weight',
     'audio_branch.layer4.1.bn1.weight',
     'audio_branch.layer4.1.bn1.bias',
     'audio_branch.layer4.1.bn1.running_mean',
     'audio_branch.layer4.1.bn1.running_var',
     'audio_branch.layer4.1.bn1.num_batches_tracked',
     'audio_branch.layer4.1.conv2.weight',
     'audio_branch.layer4.1.bn2.weight',
     'audio_branch.layer4.1.bn2.bias',
     'audio_branch.layer4.1.bn2.running_mean',
     'audio_branch.layer4.1.bn2.running_var',
     'audio_branch.layer4.1.bn2.num_batches_tracked',
     'visual_branch.init_stage.conv1.weight',
     'visual_branch.init_stage.bn1.weight',
     'visual_branch.init_stage.bn1.bias',
     'visual_branch.init_stage.bn1.running_mean',
     'visual_branch.init_stage.bn1.running_var',
     'visual_branch.init_stage.bn1.num_batches_tracked',
     'visual_branch.layer1.0.conv1.weight',
     'visual_branch.layer1.0.bn1.weight',
     'visual_branch.layer1.0.bn1.bias',
     'visual_branch.layer1.0.bn1.running_mean',
     'visual_branch.layer1.0.bn1.running_var',
     'visual_branch.layer1.0.bn1.num_batches_tracked',
     'visual_branch.layer1.0.conv2.weight',
     'visual_branch.layer1.0.bn2.weight',
     'visual_branch.layer1.0.bn2.bias',
     'visual_branch.layer1.0.bn2.running_mean',
     'visual_branch.layer1.0.bn2.running_var',
     'visual_branch.layer1.0.bn2.num_batches_tracked',
     'visual_branch.layer1.1.conv1.weight',
     'visual_branch.layer1.1.bn1.weight',
     'visual_branch.layer1.1.bn1.bias',
     'visual_branch.layer1.1.bn1.running_mean',
     'visual_branch.layer1.1.bn1.running_var',
     'visual_branch.layer1.1.bn1.num_batches_tracked',
     'visual_branch.layer1.1.conv2.weight',
     'visual_branch.layer1.1.bn2.weight',
     'visual_branch.layer1.1.bn2.bias',
     'visual_branch.layer1.1.bn2.running_mean',
     'visual_branch.layer1.1.bn2.running_var',
     'visual_branch.layer1.1.bn2.num_batches_tracked',
     'visual_branch.layer2.0.conv1.weight',
     'visual_branch.layer2.0.bn1.weight',
     'visual_branch.layer2.0.bn1.bias',
     'visual_branch.layer2.0.bn1.running_mean',
     'visual_branch.layer2.0.bn1.running_var',
     'visual_branch.layer2.0.bn1.num_batches_tracked',
     'visual_branch.layer2.0.conv2.weight',
     'visual_branch.layer2.0.bn2.weight',
     'visual_branch.layer2.0.bn2.bias',
     'visual_branch.layer2.0.bn2.running_mean',
     'visual_branch.layer2.0.bn2.running_var',
     'visual_branch.layer2.0.bn2.num_batches_tracked',
     'visual_branch.layer2.0.downsample.0.weight',
     'visual_branch.layer2.0.downsample.1.weight',
     'visual_branch.layer2.0.downsample.1.bias',
     'visual_branch.layer2.0.downsample.1.running_mean',
     'visual_branch.layer2.0.downsample.1.running_var',
     'visual_branch.layer2.0.downsample.1.num_batches_tracked',
     'visual_branch.layer2.1.conv1.weight',
     'visual_branch.layer2.1.bn1.weight',
     'visual_branch.layer2.1.bn1.bias',
     'visual_branch.layer2.1.bn1.running_mean',
     'visual_branch.layer2.1.bn1.running_var',
     'visual_branch.layer2.1.bn1.num_batches_tracked',
     'visual_branch.layer2.1.conv2.weight',
     'visual_branch.layer2.1.bn2.weight',
     'visual_branch.layer2.1.bn2.bias',
     'visual_branch.layer2.1.bn2.running_mean',
     'visual_branch.layer2.1.bn2.running_var',
     'visual_branch.layer2.1.bn2.num_batches_tracked',
     'visual_branch.layer3.0.conv1.weight',
     'visual_branch.layer3.0.bn1.weight',
     'visual_branch.layer3.0.bn1.bias',
     'visual_branch.layer3.0.bn1.running_mean',
     'visual_branch.layer3.0.bn1.running_var',
     'visual_branch.layer3.0.bn1.num_batches_tracked',
     'visual_branch.layer3.0.conv2.weight',
     'visual_branch.layer3.0.bn2.weight',
     'visual_branch.layer3.0.bn2.bias',
     'visual_branch.layer3.0.bn2.running_mean',
     'visual_branch.layer3.0.bn2.running_var',
     'visual_branch.layer3.0.bn2.num_batches_tracked',
     'visual_branch.layer3.0.downsample.0.weight',
     'visual_branch.layer3.0.downsample.1.weight',
     'visual_branch.layer3.0.downsample.1.bias',
     'visual_branch.layer3.0.downsample.1.running_mean',
     'visual_branch.layer3.0.downsample.1.running_var',
     'visual_branch.layer3.0.downsample.1.num_batches_tracked',
     'visual_branch.layer3.1.conv1.weight',
     'visual_branch.layer3.1.bn1.weight',
     'visual_branch.layer3.1.bn1.bias',
     'visual_branch.layer3.1.bn1.running_mean',
     'visual_branch.layer3.1.bn1.running_var',
     'visual_branch.layer3.1.bn1.num_batches_tracked',
     'visual_branch.layer3.1.conv2.weight',
     'visual_branch.layer3.1.bn2.weight',
     'visual_branch.layer3.1.bn2.bias',
     'visual_branch.layer3.1.bn2.running_mean',
     'visual_branch.layer3.1.bn2.running_var',
     'visual_branch.layer3.1.bn2.num_batches_tracked',
     'visual_branch.layer4.0.conv1.weight',
     'visual_branch.layer4.0.bn1.weight',
     'visual_branch.layer4.0.bn1.bias',
     'visual_branch.layer4.0.bn1.running_mean',
     'visual_branch.layer4.0.bn1.running_var',
     'visual_branch.layer4.0.bn1.num_batches_tracked',
     'visual_branch.layer4.0.conv2.weight',
     'visual_branch.layer4.0.bn2.weight',
     'visual_branch.layer4.0.bn2.bias',
     'visual_branch.layer4.0.bn2.running_mean',
     'visual_branch.layer4.0.bn2.running_var',
     'visual_branch.layer4.0.bn2.num_batches_tracked',
     'visual_branch.layer4.0.downsample.0.weight',
     'visual_branch.layer4.0.downsample.1.weight',
     'visual_branch.layer4.0.downsample.1.bias',
     'visual_branch.layer4.0.downsample.1.running_mean',
     'visual_branch.layer4.0.downsample.1.running_var',
     'visual_branch.layer4.0.downsample.1.num_batches_tracked',
     'visual_branch.layer4.1.conv1.weight',
     'visual_branch.layer4.1.bn1.weight',
     'visual_branch.layer4.1.bn1.bias',
     'visual_branch.layer4.1.bn1.running_mean',
     'visual_branch.layer4.1.bn1.running_var',
     'visual_branch.layer4.1.bn1.num_batches_tracked',
     'visual_branch.layer4.1.conv2.weight',
     'visual_branch.layer4.1.bn2.weight',
     'visual_branch.layer4.1.bn2.bias',
     'visual_branch.layer4.1.bn2.running_mean',
     'visual_branch.layer4.1.bn2.running_var',
     'visual_branch.layer4.1.bn2.num_batches_tracked',
     'linear.weight',
     'linear.bias'])
'''
