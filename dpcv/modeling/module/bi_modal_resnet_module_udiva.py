import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
# consider add pre-train weight
# model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}


def vis_conv3x3_udiva(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False) # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')


def vis_conv1x1_udiva(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def aud_conv1x9_udiva(in_planes, out_planes, stride=1):
    """1x9 convolution with padding"""
    if stride == 1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 9), stride=1, padding=(0, 4), bias=False)
    elif stride == 2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 9), stride=(1, 2*stride), padding=(0, 4), bias=False)
    else:
        raise ValueError("wrong stride value")


def aud_conv1x1_udiva(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    if stride == 1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
    elif stride == 2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=(1, 2 * stride), bias=False)
    else:
        raise ValueError("wrong stride value")


class VisInitStageUdiva(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(VisInitStageUdiva, self).__init__()
        # print('[dpcv/modeling/module/bi_modal_resnet_module_udiva.py] VisInitStage类: in_channels: ', in_channels, 'out_channels: ', out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class AudInitStageUdiva(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(AudInitStageUdiva, self).__init__()
        # print('[dpcv/modeling/module/bi_modal_resnet_module_udiva.py] AudInitStage类: 执行初始化函数__init__: in_channels: ', in_channels, 'out_channels: ', out_channels) # in_channels:  2 out_channels:  32
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 49), stride=(1, 4), padding=(0, 24), bias=False)
            # in_channels: Number of channels in the input  输入音频的通道数=2
            # out_channels: Number of channels produced by the convolution 卷积产生的通道数=32
            # kernel_size: Size of the convolving kernel 卷积核的大小=(1, 49),  卷积核的宽度为49, 卷积核的高度为1，可以设为1个int型数或者一个(int, int)型的元组。例如(2,3)是高2宽3卷积核
            # stride: Stride of the convolution 卷积步长=(1, 4), 可以设为1个int型数或者一个(int, int)型的元组。例如(2,3)是高2宽3步长
            # padding: Zero-padding added to both sides of the input 卷积核的填充=(0, 24), 可以设为1个int型数或者一个(int, int)型的元组。例如(2,3)是高2宽3填充
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 9), stride=(1, 4), padding=(0, 4))
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class BiModalBasicBlockUdiva(nn.Module):
    """
    build visual and audio conv block for resnet18 architecture
    """
    expansion = 1

    def __init__(self, conv_type, inplanes, planes, stride=1, downsample=None):
        super(BiModalBasicBlockUdiva, self).__init__()
        self.conv1 = conv_type(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_type(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class AudioVisualResNetUdiva(nn.Module):

    def __init__(self, in_channels, init_stage, block, conv,
                 channels=[64, 128, 256, 512],  # default resnet stage channel settings
                 layers=[2, 2, 2, 2],  # default resnet18 layers setting
                 out_spatial=(1, 1),
                 zero_init_residual=False,
                 branch_type='audio'):
        # print('[dpcv/modeling/networks/bi_modal_resnet_module_udiva.py] class AudioVisualResNetUdiva 开始执行构造函数 init...  所有初始化的参数: in_channels: ', in_channels, 'init_stage: ', init_stage, 'block: ', block, 'conv: ', conv, 'channels: ', channels, 'layers: ', layers, 'out_spatial: ', out_spatial, 'zero_init_residual: ', zero_init_residual)
        super(AudioVisualResNetUdiva, self).__init__() # init the super class, nn.Module, to get all the methods, attributes, etc. of nn.Module, and then add more attributes and methods, such as self.init_stage, self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool

        assert init_stage.__name__ in ["AudInitStageUdiva", "VisInitStageUdiva"], \
            "init conv stage should be 'AudInitStageUdiva' or 'VisInitStageUdiva'"
        assert len(conv) == 2, "conv should be a list containing <conv3x3 conv1x1> or <conv1x9, conv1x1> function"

        self.inplanes = channels[0] # 32
        self.conv_3x3 = conv[0] # vis_conv3x3 or aud_conv1x9
        self.conv_1x1 = conv[1] # vis_conv1x1 or aud_conv1x1
        self.init_stage = init_stage(in_channels, channels[0]) # init_stage = VisInitStageUdiva(in_channels=32, out_channels=64)
        self.layer1 = self._make_layer(block, channels[0], layers[0]) # block = BiModalBasicBlockUdiva (visual and audio conv block for resnet18 architecture), channels[0] = 32, layers[0] = 2, so self.layer1 = self._make_layer(BiModalBasicBlockUdiva, 32, 2)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2) # block = BiModalBasicBlockUdiva, channels[1] = 64, layers[1] = 2, stride=2, so self.layer2 = self._make_layer(BiModalBasicBlockUdiva, 64, 2, stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2) # block = BiModalBasicBlockUdiva, channels[2] = 128, layers[2] = 2, stride=2, so self.layer3 = self._make_layer(BiModalBasicBlockUdiva, 128, 2, stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2) # block = BiModalBasicBlockUdiva, channels[3] = 256, layers[3] = 2, stride=2, so self.layer4 = self._make_layer(BiModalBasicBlockUdiva, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(out_spatial) # out_spatial = (1, 1), so self.avgpool = nn.AdaptiveAvgPool2d((1, 1)), which is a 2D adaptive average pooling layer, which means that the output size is (1, 1), and the input size can be any size, and the output size is the average of the input size, so the output size is (1, 1)
        self.branch_type = branch_type
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=3) # input_size表示输入的特征维度(embedding_size)，hidden_size表示隐藏层的维度，num_layers表示LSTM的层数
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,  block, planes, blocks, stride=1): #  _make_layer is a function to build a layer, which contains several blocks, and the number of blocks is defined by the parameter layers
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: # stride=2, inplanes=32, planes=32,64,128,256, block.expansion = 1, so if 2 != 1 or 32 != planes * 1, then downsample = nn.Sequential(
            downsample = nn.Sequential( # downsample means the downsample layer, which is used to downsample the input, so that the input and the output of the block can be added together, so that the input and the output of the block have the same size, so that the input and the output of the block can be added together
                self.conv_1x1(self.inplanes, planes * block.expansion, stride), # inplanes=32, planes=32,64,128,256, block.expansion = 1, so self.conv_1x1(32, 32/64/128/256, 2)
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.conv_3x3, self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.conv_3x3, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): # forward will be called when you call the model, so the input x is the input of the model, and the output is the output of the model
        # print('[dpcv/modeling/module/bi_modal_resnet_module_udiva.py] - 模型的输入数据 x.shape = ', x.shape)
        # 输入的数据维度 audio: x.shape =  torch.Size([8, 2, 1, 50176])=[batchsize, fc1+fc2=2, 1, 50176] or visual: torch.Size([8, 16, 6, 224, 224]) = [batchsize, sample_size=16frames, 3*2(fc1+fc2)=6, 224, 224]
        if self.branch_type == 'audio': # 音频文件因为没有16帧这一个维度（包含在最后一维度里面了:256000=16秒*采样率16000），和img不一样，所以不需要输入LSTM网络。
            # print('[dpcv/modeling/module/bi_modal_resnet_module_udiva.py] - class AudioVisualResNetUdiva(nn.Module): 正在执行forward逻辑, 输入的数据维度 x.shape = ', x.shape)  # audio: x.shape =  torch.Size([batchsize, fc1+fc2=2, 1, 256000=16秒*采样率16000])
            x = self.init_stage(x) # 经过init_stage后的数据维度 audio: x.shape =  torch.Size([8, 32, 1, 3136]) or visual: torch.Size([8, 32, 56, 56])
            # print('[dpcv/modeling/module/bi_modal_resnet_module_udiva.py] - class AudioVisualResNetUdiva(nn.Module): 正在执行forward逻辑, 经过init_stage后的数据维度 x.shape = ', x.shape)
            x = self.layer1(x) # 经过 layer1 后的数据维度 x.shape =  torch.Size([8, 32, 1, 3136]) or torch.Size([8, 32, 56, 56])
            # print('[dpcv/modeling/module/bi_modal_resnet_module_udiva.py] - class AudioVisualResNetUdiva(nn.Module): 正在执行forward逻辑, 经过 layer1 后的数据维度 x.shape = ', x.shape)
            x = self.layer2(x) # 经过 layer2 后的数据维度 x.shape =  torch.Size([8, 64, 1, 784]) or torch.Size([8, 64, 28, 28])
            # print('[dpcv/modeling/module/bi_modal_resnet_module_udiva.py] - class AudioVisualResNetUdiva(nn.Module): 正在执行forward逻辑, 经过 layer2 后的数据维度 x.shape = ', x.shape)
            x = self.layer3(x) # 经过 layer3 后的数据维度 x.shape =  torch.Size([8, 128, 1, 196]) or torch.Size([8, 128, 14, 14])
            # print('[dpcv/modeling/module/bi_modal_resnet_module_udiva.py] - class AudioVisualResNetUdiva(nn.Module): 正在执行forward逻辑, 经过 layer3 后的数据维度 x.shape = ', x.shape)
            x = self.layer4(x) # 经过 layer4 后的数据维度 x.shape =  torch.Size([8, 256, 1, 49]) or torch.Size([8, 256, 7, 7])
            # print('[dpcv/modeling/module/bi_modal_resnet_module_udiva.py] - class AudioVisualResNetUdiva(nn.Module): 正在执行forward逻辑, 经过 layer4 后的数据维度 x.shape = ', x.shape)
            x = self.avgpool(x) # 经过 avgpool 后的数据维度（即最终输出的数据维度） x.shape =  torch.Size([8, 256, 1, 1]) or torch.Size([8, 256, 1, 1])
            # print('[dpcv/modeling/module/bi_modal_resnet_module_udiva.py] - class AudioVisualResNetUdiva(nn.Module): 正在执行forward逻辑, 经过 avgpool 后的数据维度（即最终输出的数据维度） x.shape = ', x.shape)
        elif self.branch_type == 'visual':
            hidden = None
            for frame_id in range(x.size(1)):
                with torch.no_grad():
                    x_frame_feature = self.init_stage(x[:, frame_id, :, :, :])
                    x_frame_feature = self.layer1(x_frame_feature)
                    x_frame_feature = self.layer2(x_frame_feature)
                    x_frame_feature = self.layer3(x_frame_feature)
                    x_frame_feature = self.layer4(x_frame_feature)
                    x_frame_feature = self.avgpool(x_frame_feature)
                    # print('[dpcv/modeling/module/bi_modal_resnet_module_udiva.py] - class AudioVisualResNetUdiva(nn.Module): x_frame_feature.shape = ', x_frame_feature.shape, ' x_frame_feature.unsqueeze(0).shape = ', x_frame_feature.unsqueeze(0).shape) # x_frame_feature.shape =  torch.Size([batch_size, 256, 1, 1])  x_frame_feature.unsqueeze(0).shape =  torch.Size([1, batch_size, 256, 1, 1])
                x_frame_feature = x_frame_feature.view(x_frame_feature.size(0), -1) # 将x_frame_feature的维度从 torch.Size([batch_size, 256, 1, 1]) 转换为 torch.Size([batch_size, 256])
                # print('[dpcv/modeling/module/bi_modal_resnet_module_udiva.py] - class AudioVisualResNetUdiva(nn.Module): after x_frame_feature.shape = ', x_frame_feature.shape) # after x_frame_feature.shape = torch.Size([batch_size, 256])
                out, hidden = self.lstm(x_frame_feature.unsqueeze(0), hidden) # LSTM forward函数的第一个参数inputs的各个参数为：1、seq_len=1 2、batch_size 3、input_size=embedding_size=256
                # print('[dpcv/modeling/module/bi_modal_resnet_module_udiva.py] - class AudioVisualResNetUdiva(nn.Module): out.shape = ', out.shape, ' hidden[0].shape = ', hidden[0].shape) # out.shape =  torch.Size([1, 2=batch_size, 256])  hidden[0].shape =  torch.Size([3, 2=batch_size, 256])
            x = out[-1, :, :] # -1 表示最后一个序列，即最后一个时间步的输出
            # print('[dpcv/modeling/module/bi_modal_resnet_module_udiva.py] - class AudioVisualResNetUdiva(nn.Module): final x.shape = ', x.shape) # final x.shape =  torch.Size([batch_size, 256])
        return x
