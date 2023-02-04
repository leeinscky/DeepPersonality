# 原文链接：https://blog.csdn.net/hugvgngj/article/details/111186170

import torch  # 命令行是逐行立即执行的
content = torch.load('resnet18-5c106cde.pth')
# print(content.keys())   # keys()

# 之后有其他需求比如要看 key 为 model 的内容有啥
# print(content['conv1.weight'].shape) # torch.Size([64, 3, 7, 7])

for key in content.keys():
    print(f'key:{key}, \t \t \t \t{content[key].shape}') # 打印结果见下方


# print(content.keys()) 打印结果
'''
odict_keys(
    ['conv1.weight',
     'bn1.running_mean',
     'bn1.running_var',
     'bn1.weight',
     'bn1.bias',
     'layer1.0.conv1.weight',
     'layer1.0.bn1.running_mean',
     'layer1.0.bn1.running_var',
     'layer1.0.bn1.weight',
     'layer1.0.bn1.bias',
     'layer1.0.conv2.weight',
     'layer1.0.bn2.running_mean',
     'layer1.0.bn2.running_var',
     'layer1.0.bn2.weight',
     'layer1.0.bn2.bias',
     'layer1.1.conv1.weight',
     'layer1.1.bn1.running_mean',
     'layer1.1.bn1.running_var',
     'layer1.1.bn1.weight',
     'layer1.1.bn1.bias',
     'layer1.1.conv2.weight',
     'layer1.1.bn2.running_mean',
     'layer1.1.bn2.running_var',
     'layer1.1.bn2.weight',
     'layer1.1.bn2.bias',
     'layer2.0.conv1.weight',
     'layer2.0.bn1.running_mean',
     'layer2.0.bn1.running_var',
     'layer2.0.bn1.weight',
     'layer2.0.bn1.bias',
     'layer2.0.conv2.weight',
     'layer2.0.bn2.running_mean',
     'layer2.0.bn2.running_var',
     'layer2.0.bn2.weight',
     'layer2.0.bn2.bias',
     'layer2.0.downsample.0.weight',
     'layer2.0.downsample.1.running_mean',
     'layer2.0.downsample.1.running_var',
     'layer2.0.downsample.1.weight',
     'layer2.0.downsample.1.bias',
     'layer2.1.conv1.weight',
     'layer2.1.bn1.running_mean',
     'layer2.1.bn1.running_var',
     'layer2.1.bn1.weight',
     'layer2.1.bn1.bias',
     'layer2.1.conv2.weight',
     'layer2.1.bn2.running_mean',
     'layer2.1.bn2.running_var',
     'layer2.1.bn2.weight',
     'layer2.1.bn2.bias',
     'layer3.0.conv1.weight',
     'layer3.0.bn1.running_mean',
     'layer3.0.bn1.running_var',
     'layer3.0.bn1.weight',
     'layer3.0.bn1.bias',
     'layer3.0.conv2.weight',
     'layer3.0.bn2.running_mean',
     'layer3.0.bn2.running_var',
     'layer3.0.bn2.weight',
     'layer3.0.bn2.bias',
     'layer3.0.downsample.0.weight',
     'layer3.0.downsample.1.running_mean',
     'layer3.0.downsample.1.running_var',
     'layer3.0.downsample.1.weight',
     'layer3.0.downsample.1.bias',
     'layer3.1.conv1.weight',
     'layer3.1.bn1.running_mean',
     'layer3.1.bn1.running_var',
     'layer3.1.bn1.weight',
     'layer3.1.bn1.bias',
     'layer3.1.conv2.weight',
     'layer3.1.bn2.running_mean',
     'layer3.1.bn2.running_var',
     'layer3.1.bn2.weight',
     'layer3.1.bn2.bias',
     'layer4.0.conv1.weight',
     'layer4.0.bn1.running_mean',
     'layer4.0.bn1.running_var',
     'layer4.0.bn1.weight',
     'layer4.0.bn1.bias',
     'layer4.0.conv2.weight',
     'layer4.0.bn2.running_mean',
     'layer4.0.bn2.running_var',
     'layer4.0.bn2.weight',
     'layer4.0.bn2.bias',
     'layer4.0.downsample.0.weight',
     'layer4.0.downsample.1.running_mean',
     'layer4.0.downsample.1.running_var',
     'layer4.0.downsample.1.weight',
     'layer4.0.downsample.1.bias',
     'layer4.1.conv1.weight',
     'layer4.1.bn1.running_mean',
     'layer4.1.bn1.running_var',
     'layer4.1.bn1.weight',
     'layer4.1.bn1.bias',
     'layer4.1.conv2.weight',
     'layer4.1.bn2.running_mean',
     'layer4.1.bn2.running_var',
     'layer4.1.bn2.weight',
     'layer4.1.bn2.bias',
     'fc.weight',
     'fc.bias']
    )
'''

# print(f'key:{key}, {content[key].shape}') 打印结果
''' 
key:conv1.weight, 	 	 	 torch.Size([64, 3, 7, 7])
key:bn1.running_mean, 	 	 	 torch.Size([64])
key:bn1.running_var, 	 	 	 torch.Size([64])
key:bn1.weight, 	 	 	 torch.Size([64])
key:bn1.bias, 	 	 	 torch.Size([64])
key:layer1.0.conv1.weight, 	 	 	 torch.Size([64, 64, 3, 3])
key:layer1.0.bn1.running_mean, 	 	 	 torch.Size([64])
key:layer1.0.bn1.running_var, 	 	 	 torch.Size([64])
key:layer1.0.bn1.weight, 	 	 	 torch.Size([64])
key:layer1.0.bn1.bias, 	 	 	 torch.Size([64])
key:layer1.0.conv2.weight, 	 	 	 torch.Size([64, 64, 3, 3])
key:layer1.0.bn2.running_mean, 	 	 	 torch.Size([64])
key:layer1.0.bn2.running_var, 	 	 	 torch.Size([64])
key:layer1.0.bn2.weight, 	 	 	 torch.Size([64])
key:layer1.0.bn2.bias, 	 	 	 torch.Size([64])
key:layer1.1.conv1.weight, 	 	 	 torch.Size([64, 64, 3, 3])
key:layer1.1.bn1.running_mean, 	 	 	 torch.Size([64])
key:layer1.1.bn1.running_var, 	 	 	 torch.Size([64])
key:layer1.1.bn1.weight, 	 	 	 torch.Size([64])
key:layer1.1.bn1.bias, 	 	 	 torch.Size([64])
key:layer1.1.conv2.weight, 	 	 	 torch.Size([64, 64, 3, 3])
key:layer1.1.bn2.running_mean, 	 	 	 torch.Size([64])
key:layer1.1.bn2.running_var, 	 	 	 torch.Size([64])
key:layer1.1.bn2.weight, 	 	 	 torch.Size([64])
key:layer1.1.bn2.bias, 	 	 	 torch.Size([64])
key:layer2.0.conv1.weight, 	 	 	 torch.Size([128, 64, 3, 3])
key:layer2.0.bn1.running_mean, 	 	 	 torch.Size([128])
key:layer2.0.bn1.running_var, 	 	 	 torch.Size([128])
key:layer2.0.bn1.weight, 	 	 	 torch.Size([128])
key:layer2.0.bn1.bias, 	 	 	 torch.Size([128])
key:layer2.0.conv2.weight, 	 	 	 torch.Size([128, 128, 3, 3])
key:layer2.0.bn2.running_mean, 	 	 	 torch.Size([128])
key:layer2.0.bn2.running_var, 	 	 	 torch.Size([128])
key:layer2.0.bn2.weight, 	 	 	 torch.Size([128])
key:layer2.0.bn2.bias, 	 	 	 torch.Size([128])
key:layer2.0.downsample.0.weight, 	 	 	 torch.Size([128, 64, 1, 1])
key:layer2.0.downsample.1.running_mean, 	 	 	 torch.Size([128])
key:layer2.0.downsample.1.running_var, 	 	 	 torch.Size([128])
key:layer2.0.downsample.1.weight, 	 	 	 torch.Size([128])
key:layer2.0.downsample.1.bias, 	 	 	 torch.Size([128])
key:layer2.1.conv1.weight, 	 	 	 torch.Size([128, 128, 3, 3])
key:layer2.1.bn1.running_mean, 	 	 	 torch.Size([128])
key:layer2.1.bn1.running_var, 	 	 	 torch.Size([128])
key:layer2.1.bn1.weight, 	 	 	 torch.Size([128])
key:layer2.1.bn1.bias, 	 	 	 torch.Size([128])
key:layer2.1.conv2.weight, 	 	 	 torch.Size([128, 128, 3, 3])
key:layer2.1.bn2.running_mean, 	 	 	 torch.Size([128])
key:layer2.1.bn2.running_var, 	 	 	 torch.Size([128])
key:layer2.1.bn2.weight, 	 	 	 torch.Size([128])
key:layer2.1.bn2.bias, 	 	 	 torch.Size([128])
key:layer3.0.conv1.weight, 	 	 	 torch.Size([256, 128, 3, 3])
key:layer3.0.bn1.running_mean, 	 	 	 torch.Size([256])
key:layer3.0.bn1.running_var, 	 	 	 torch.Size([256])
key:layer3.0.bn1.weight, 	 	 	 torch.Size([256])
key:layer3.0.bn1.bias, 	 	 	 torch.Size([256])
key:layer3.0.conv2.weight, 	 	 	 torch.Size([256, 256, 3, 3])
key:layer3.0.bn2.running_mean, 	 	 	 torch.Size([256])
key:layer3.0.bn2.running_var, 	 	 	 torch.Size([256])
key:layer3.0.bn2.weight, 	 	 	 torch.Size([256])
key:layer3.0.bn2.bias, 	 	 	 torch.Size([256])
key:layer3.0.downsample.0.weight, 	 	 	 torch.Size([256, 128, 1, 1])
key:layer3.0.downsample.1.running_mean, 	 	 	 torch.Size([256])
key:layer3.0.downsample.1.running_var, 	 	 	 torch.Size([256])
key:layer3.0.downsample.1.weight, 	 	 	 torch.Size([256])
key:layer3.0.downsample.1.bias, 	 	 	 torch.Size([256])
key:layer3.1.conv1.weight, 	 	 	 torch.Size([256, 256, 3, 3])
key:layer3.1.bn1.running_mean, 	 	 	 torch.Size([256])
key:layer3.1.bn1.running_var, 	 	 	 torch.Size([256])
key:layer3.1.bn1.weight, 	 	 	 torch.Size([256])
key:layer3.1.bn1.bias, 	 	 	 torch.Size([256])
key:layer3.1.conv2.weight, 	 	 	 torch.Size([256, 256, 3, 3])
key:layer3.1.bn2.running_mean, 	 	 	 torch.Size([256])
key:layer3.1.bn2.running_var, 	 	 	 torch.Size([256])
key:layer3.1.bn2.weight, 	 	 	 torch.Size([256])
key:layer3.1.bn2.bias, 	 	 	 torch.Size([256])
key:layer4.0.conv1.weight, 	 	 	 torch.Size([512, 256, 3, 3])
key:layer4.0.bn1.running_mean, 	 	 	 torch.Size([512])
key:layer4.0.bn1.running_var, 	 	 	 torch.Size([512])
key:layer4.0.bn1.weight, 	 	 	 torch.Size([512])
key:layer4.0.bn1.bias, 	 	 	 torch.Size([512])
key:layer4.0.conv2.weight, 	 	 	 torch.Size([512, 512, 3, 3])
key:layer4.0.bn2.running_mean, 	 	 	 torch.Size([512])
key:layer4.0.bn2.running_var, 	 	 	 torch.Size([512])
key:layer4.0.bn2.weight, 	 	 	 torch.Size([512])
key:layer4.0.bn2.bias, 	 	 	 torch.Size([512])
key:layer4.0.downsample.0.weight, 	 	 	 torch.Size([512, 256, 1, 1])
key:layer4.0.downsample.1.running_mean, 	 	 	 torch.Size([512])
key:layer4.0.downsample.1.running_var, 	 	 	 torch.Size([512])
key:layer4.0.downsample.1.weight, 	 	 	 torch.Size([512])
key:layer4.0.downsample.1.bias, 	 	 	 torch.Size([512])
key:layer4.1.conv1.weight, 	 	 	 torch.Size([512, 512, 3, 3])
key:layer4.1.bn1.running_mean, 	 	 	 torch.Size([512])
key:layer4.1.bn1.running_var, 	 	 	 torch.Size([512])
key:layer4.1.bn1.weight, 	 	 	 torch.Size([512])
key:layer4.1.bn1.bias, 	 	 	 torch.Size([512])
key:layer4.1.conv2.weight, 	 	 	 torch.Size([512, 512, 3, 3])
key:layer4.1.bn2.running_mean, 	 	 	 torch.Size([512])
key:layer4.1.bn2.running_var, 	 	 	 torch.Size([512])
key:layer4.1.bn2.weight, 	 	 	 torch.Size([512])
key:layer4.1.bn2.bias, 	 	 	 torch.Size([512])
key:fc.weight, 	 	 	 torch.Size([1000, 512])
key:fc.bias, 	 	 	 torch.Size([1000])
'''