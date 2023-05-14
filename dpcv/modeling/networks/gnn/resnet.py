# ResNet
# Deep Residual Learning for Image Recognition
# Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun

import os
import math
import torch
import torch.nn as nn
import torchvision.models
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# you need to download the models to ~/.torch/models
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }

# models_dir = os.path.expanduser('pre_trained_weights/resnet')
models_dir = os.path.expanduser('pre_trained_weights/ME-GraphAU/resnet')
model_name = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channels, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # self.channels = 3 # image channel is 3 (original)
        # self.channels = 6 # image channel is 6 (a pair of video frames)
        # self.channels = 2 # audio channel is 2 (a pair of audio frames)
        self.channels = in_channels
        self.conv1 = nn.Conv2d(self.channels, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        print('[ResNet] input x shape: ', x.shape) # 3 channels: [bs, 3, 112, 112]; 6 channels: [bs, 6, 112, 112]
        x = self.conv1(x)
        print('[ResNet] after conv1, x shape: ', x.shape) # 3 channels: [bs, 64, 56, 56]; 6 channels: [bs, 64, 56, 56]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print('[ResNet] after maxpool, x shape: ', x.shape) # 3 channels: [bs, 64, 28, 28]; 6 channels: [bs, 64, 28, 28]

        x = self.layer1(x)
        print('[ResNet] after layer1, x shape: ', x.shape) # 3 channels: [bs, 256, 28, 28]; 6 channels: [bs, 256, 28, 28]
        x = self.layer2(x)
        print('[ResNet] after layer2, x shape: ', x.shape) # 3 channels: [bs, 512, 14, 14]
        x = self.layer3(x)
        print('[ResNet] after layer3, x shape: ', x.shape) # 3 channels: [bs, 1024, 7, 7]
        x = self.layer4(x)
        print('[ResNet] after layer4, x shape: ', x.shape) # 3 channels: [bs, 2048, 4, 4]

        b,c,h,w = x.shape
        print('[ResNet] after layers, x shape: ', x.shape) # 3 channels: [bs, 2048, 4, 4]; 6 channels: [bs, 2048, 4, 4]
        
        x = x.view(b,c,-1).permute(0,2,1) # 3 channels: [bs, 2048, 4, 4] -> [bs, 2048, 16] -> [bs, 16, 2048]
        print('[ResNet] output x shape: ', x.shape) # 3 channels: [bs, 16, 2048]; 6 channels: [bs, 16, 2048]
        
        return x


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet18'])))
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet34'])))
    return model


def resnet50(pretrained=True, in_channels=3, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], in_channels, **kwargs)
    if pretrained:
        print('[resnet.py] loading pretrained ResNet model: ', os.path.join(models_dir, model_name['resnet50']))
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet50']), map_location=device))
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet101'])))
        print('[resnet.py] loading pretrained ResNet model: ', os.path.join(models_dir, model_name['resnet101']))
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet101']), map_location=device))
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet152'])))
    return model
