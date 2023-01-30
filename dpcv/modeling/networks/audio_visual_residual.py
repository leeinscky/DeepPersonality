import torch
import torch.nn as nn

# 非udiva数据使用下列三个导入
from dpcv.modeling.module.bi_modal_resnet_module import AudioVisualResNet, AudInitStage
from dpcv.modeling.module.bi_modal_resnet_module import VisInitStage, BiModalBasicBlock
from dpcv.modeling.module.bi_modal_resnet_module import aud_conv1x9, aud_conv1x1, vis_conv3x3, vis_conv1x1

# udiva数据使用下列三个导入
from dpcv.modeling.module.bi_modal_resnet_module_udiva import AudioVisualResNetUdiva, AudInitStageUdiva
from dpcv.modeling.module.bi_modal_resnet_module_udiva import VisInitStageUdiva, BiModalBasicBlockUdiva
from dpcv.modeling.module.bi_modal_resnet_module_udiva import aud_conv1x9_udiva, aud_conv1x1_udiva, vis_conv3x3_udiva, vis_conv1x1_udiva

from dpcv.modeling.module.weight_init_helper import initialize_weights
from dpcv.modeling.networks.build import NETWORK_REGISTRY


class AudioVisualResNet18(nn.Module): # nn.Module是所有神经网络模块的基类，你的网络也应该继承这个类，这样可以方便地使用各种神经网络模块，比如卷积、全连接、激活函数等等，这些模块都是nn.Module的子类。你可以直接调用，也可以自己定义，然后继承nn.Module，这样就可以使用了，这样做的好处是可以方便地组合各种模块，构建复杂的网络结构，而且可以方便地调用各种模块的参数，比如权重、偏置等等。

    def __init__(self, init_weights=True, return_feat=False):
        super(AudioVisualResNet18, self).__init__() # super函数是用来调用父类的构造函数的，这里的AudioVisualResNet18是继承了nn.Module类的，所以这里的super函数就是调用了nn.Module类的构造函数，这里的init_weights表示的是是否初始化网络的权重，return_feat表示的是是否返回特征
        self.return_feature = return_feat
        self.audio_branch = AudioVisualResNet(
            in_channels=1, init_stage=AudInitStage, # in_channels=1表示输入的通道数为1，init_stage=AudInitStage表示使用AudInitStage这个类作为初始化层
            block=BiModalBasicBlock, conv=[aud_conv1x9, aud_conv1x1], # block是一个类，conv是一个函数
            channels=[32, 64, 128, 256], # 32, 64, 128, 256 are the number of channels in each layer, which is the number of filters in each layer, 这里的32, 64, 128, 256是每一层的卷积核的个数，也就是每一层的通道数
            layers=[2, 2, 2, 2] # 2, 2, 2, 2 are the number of layers in each layer, 这里的2, 2, 2, 2是每一层的卷积层数
        )
        self.visual_branch = AudioVisualResNet(
            in_channels=3, init_stage=VisInitStage,
            block=BiModalBasicBlock, conv=[vis_conv3x3, vis_conv1x1],
            channels=[32, 64, 128, 256],
            layers=[2, 2, 2, 2]
        )
        self.linear = nn.Linear(512, 5)

        if init_weights:
            initialize_weights(self)

    def forward(self, aud_input, vis_input): # forward函数是必须要定义的，这个函数是用来定义网络的前向传播的，这里的aud_input和vis_input是输入的音频和视频数据, aud_input的shape是[batch_size, 1, 1, 50176], vis_input的shape是[batch_size, 3, 224, 224]
        # 将音频和视频数据分别输入到音频分支和视频分支中，得到音频分支和视频分支的输出
        # print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 音频和视频数据输入数据的维度为, aud_input.shape: ', aud_input.shape, ' vis_input.shape: ', vis_input.shape)
        aud_x = self.audio_branch(aud_input) # aud_x的shape是[batch_size, 256, 1, 1]
        vis_x = self.visual_branch(vis_input) # vis_x的shape是[batch_size, 256, 1, 1]
        # print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 音频分支和视频分支的输出数据的维度为 aud_x.shape: ', aud_x.shape, ' vis_x.shape: ', vis_x.shape)

        aud_x = aud_x.view(aud_x.size(0), -1) #view函数是用来改变tensor的形状的，这里的aud_x.size(0)表示的是batch_size, -1表示的是自动计算剩下的维度的大小，这里的aud_x的shape是[batch_size, 256]，也就是把aud_x的形状从[batch_size, 256, 1, 1]变成了[batch_size, 256]
        vis_x = vis_x.view(vis_x.size(0), -1) # 这里的vis_x的shape是[batch_size, 256]，也就是把vis_x的形状从[batch_size, 256, 1, 1]变成了[batch_size, 256]

        # print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 经过view函数之后, 音频分支和视频分支的输出数据的维度为 aud_x.shape: ', aud_x.shape, ' vis_x.shape: ', vis_x.shape)

        # 将音频分支和视频分支的输出进行拼接
        feat = torch.cat([aud_x, vis_x], dim=-1) # torch.cat是用来拼接tensor的，这里的dim=-1表示的是拼接的维度，这里的feat的shape是[batch_size, 512]，也就是把aud_x和vis_x拼接在了一起
        # print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 经过torch.cat函数之后即将音频分支和视频分支的输出进行拼接 拼接后得到的feat维度为 feat.shape: ', feat.shape)
        x = self.linear(feat) # self.linear = nn.Linear(512, 5) 这里的x的shape是[batch_size, 5]，也就是把feat输入到全连接层中，全连接层的输出是5个类别的概率，这里的5表示的是5个类别。
        # print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 将音频分支和视频分支的输出进行拼接之后的数据feat, 输入到全连接层中, 得到的输出数据x的维度为 x.shape: ', x.shape)
        x = torch.sigmoid(x) # 这里的x的shape是[batch_size, 5]，也就是把x输入到sigmoid函数中，sigmoid函数的作用是把x的每一个元素都压缩到0到1之间，这里的x的每一个元素都表示的是一个类别的概率。
        # print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... x经过sigmoid激活函数处理后, 得到的输出数据的维度为 x.shape: ', x.shape)
        # x = torch.tanh(x)
        # x = (x + 1) / 2  # scale tanh output to [0, 1]
        if self.return_feature: 
            return x, feat
        return x


class AudioVisualResNet18Udiva(nn.Module): # 基于UDIVA的纯CNN模型结构：没有LSTM层处理特征序列
    def __init__(self, init_weights=True, return_feat=False):
        print('[dpcv/modeling/networks/audio_visual_residual.py] class AudioVisualResNet18Udiva - 开始执行构造函数__init__... ')
        super(AudioVisualResNet18Udiva, self).__init__() # super函数是用来调用父类的构造函数的，这里的AudioVisualResNet18Udiva是继承了nn.Module类的，所以这里的super函数就是调用了nn.Module类的构造函数，这里的init_weights表示的是是否初始化网络的权重，return_feat表示的是是否返回特征
        self.return_feature = return_feat
        print('[dpcv/modeling/networks/audio_visual_residual.py] class AudioVisualResNet18Udiva - 正在执行构造函数__init__... 准备调用 AudioVisualResNetUdiva 类的构造函数, 传入的参数 in_channels=2, init_stage=AudInitStageUdiva, block=BiModalBasicBlockUdiva, conv=[aud_conv1x9_udiva, aud_conv1x1_udiva], channels=[32, 64, 128, 256], layers=[2, 2, 2, 2] ')
        self.audio_branch = AudioVisualResNetUdiva(
            in_channels=2, init_stage=AudInitStageUdiva, # in_channels=1表示输入的通道数为1，init_stage=AudInitStage表示使用AudInitStage这个类作为初始化层
            block=BiModalBasicBlockUdiva, conv=[aud_conv1x9_udiva, aud_conv1x1_udiva], # block是一个类，conv是一个函数
            channels=[32, 64, 128, 256], # 32, 64, 128, 256 are the number of channels in each layer, which is the number of filters in each layer, 这里的32, 64, 128, 256是每一层的卷积核的个数，也就是每一层的通道数
            layers=[2, 2, 2, 2] # 2, 2, 2, 2 are the number of layers in each layer, 这里的2, 2, 2, 2是每一层的卷积层数
        )
        self.visual_branch = AudioVisualResNetUdiva(
            in_channels=6, init_stage=VisInitStageUdiva,
            block=BiModalBasicBlockUdiva, conv=[vis_conv3x3_udiva, vis_conv1x1_udiva],
            channels=[32, 64, 128, 256],
            layers=[2, 2, 2, 2]
        )
        self.linear = nn.Linear(512, 2) # nn.Linear是全连接层，这里的512表示的是输入的特征维度，2表示的是输出的特征维度，2表示的是输出的类别数，也就是输出有2个维度，每个维度表示的是一个类别的概率，这里的2表示的是二分类问题，如果是多分类问题，这里的2就要改成类别的个数，比如如果是5分类问题，这里的2就要改成5。

        if init_weights:
            initialize_weights(self)
        print('[dpcv/modeling/networks/audio_visual_residual.py] class AudioVisualResNet18Udiva - 结束执行构造函数__init__... ')

    def forward(self, aud_input, vis_input): # forward函数是必须要定义的，这个函数是用来定义网络的前向传播的，这里的aud_input和vis_input是输入的音频和视频数据, aud_input的shape是[batch_size, 1, 1, 50176], vis_input的shape是[batch_size, 3, 224, 224]
        # 将音频和视频数据分别输入到音频分支和视频分支中，得到音频分支和视频分支的输出
        print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 音频和视频数据输入数据的维度为, aud_input.shape: ', aud_input.shape, ' vis_input.shape: ', vis_input.shape)
        aud_x = self.audio_branch(aud_input) # aud_x的shape是[batch_size, 256, 1, 1]
        vis_x = self.visual_branch(vis_input) # vis_x的shape是[batch_size, 256, 1, 1]
        print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 音频分支和视频分支的输出数据的维度为 aud_x.shape: ', aud_x.shape, ' vis_x.shape: ', vis_x.shape)

        aud_x = aud_x.view(aud_x.size(0), -1) #view函数是用来改变tensor的形状的，这里的aud_x.size(0)表示的是batch_size, -1表示的是自动计算剩下的维度的大小，这里的aud_x的shape是[batch_size, 256]，也就是把aud_x的形状从[batch_size, 256, 1, 1]变成了[batch_size, 256]
        vis_x = vis_x.view(vis_x.size(0), -1) # 这里的vis_x的shape是[batch_size, 256]，也就是把vis_x的形状从[batch_size, 256, 1, 1]变成了[batch_size, 256]

        print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 经过view函数之后, 音频分支和视频分支的输出数据的维度为 aud_x.shape: ', aud_x.shape, ' vis_x.shape: ', vis_x.shape)

        # 将音频分支和视频分支的输出进行cat拼接，然后输入全连接层，最后输入激活函数中得到最终输出
        feat = torch.cat([aud_x, vis_x], dim=-1) # torch.cat是用来拼接tensor的，这里的dim=-1表示的是拼接的维度，这里的feat的shape是[batch_size, 512]，也就是把aud_x和vis_x拼接在了一起
        print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 经过torch.cat函数之后即将音频分支和视频分支的输出进行拼接 拼接后得到的feat维度为 feat.shape: ', feat.shape)
        x = self.linear(feat) # self.linear = nn.Linear(512, 5) 这里的x的shape是[batch_size, 5]，也就是把feat输入到全连接层中，全连接层的输出是5个类别的概率，这里的5表示的是5个类别。全连接层的作用也就是把feat的维度从[batch_size, 512]变成了[batch_size, 5]
        print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 将音频分支和视频分支的输出进行拼接之后的数据feat, 输入到全连接层中, 得到的输出数据x的维度为 x.shape: ', x.shape, ' x=', x)
        x = torch.sigmoid(x) # 这里的x的shape是[batch_size, 5]，也就是把x输入到sigmoid函数中，sigmoid函数的作用是把x的每一个元素都压缩到0到1之间，这里的x的每一个元素都表示的是一个类别的概率。
        print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... x经过sigmoid激活函数处理后, 得到的输出数据的维度为 x.shape: ', x.shape, ' x=', x)
        # x = torch.tanh(x)
        # x = (x + 1) / 2  # scale tanh output to [0, 1]
        if self.return_feature: 
            return x, feat
        return x


class AudioVisualResNet18LSTMUdiva(nn.Module):  # 基于UDIVA的CNN-LSTM模型结构：加入了LSTM层处理特征序列
    def __init__(self, init_weights=True, return_feat=False):
        print('[dpcv/modeling/networks/audio_visual_residual.py] class AudioVisualResNet18Udiva - 开始执行构造函数__init__... ')
        super(AudioVisualResNet18LSTMUdiva, self).__init__() # super函数是用来调用父类的构造函数的，这里的AudioVisualResNet18Udiva是继承了nn.Module类的，所以这里的super函数就是调用了nn.Module类的构造函数，这里的init_weights表示的是是否初始化网络的权重，return_feat表示的是是否返回特征
        self.return_feature = return_feat
        print('[dpcv/modeling/networks/audio_visual_residual.py] class AudioVisualResNet18Udiva - 正在执行构造函数__init__... 准备调用 AudioVisualResNetUdiva 类的构造函数, 传入的参数 in_channels=2, init_stage=AudInitStageUdiva, block=BiModalBasicBlockUdiva, conv=[aud_conv1x9_udiva, aud_conv1x1_udiva], channels=[32, 64, 128, 256], layers=[2, 2, 2, 2] ')
        self.audio_branch = AudioVisualResNetUdiva(
            in_channels=2, init_stage=AudInitStageUdiva, # in_channels=1表示输入的通道数为1，init_stage=AudInitStage表示使用AudInitStage这个类作为初始化层
            block=BiModalBasicBlockUdiva, conv=[aud_conv1x9_udiva, aud_conv1x1_udiva], # block是一个类，conv是一个函数
            channels=[32, 64, 128, 256], # 32, 64, 128, 256 are the number of channels in each layer, which is the number of filters in each layer, 这里的32, 64, 128, 256是每一层的卷积核的个数，也就是每一层的通道数
            layers=[2, 2, 2, 2], # 2, 2, 2, 2 are the number of layers in each layer, 这里的2, 2, 2, 2是每一层的卷积层数
            branch_type='audio'
        )
        self.visual_branch = AudioVisualResNetUdiva(
            in_channels=6, init_stage=VisInitStageUdiva,
            block=BiModalBasicBlockUdiva, conv=[vis_conv3x3_udiva, vis_conv1x1_udiva],
            channels=[32, 64, 128, 256],
            layers=[2, 2, 2, 2],
            branch_type='visual'
        )
        self.linear = nn.Linear(512, 2) # nn.Linear是全连接层，这里的512表示的是输入的特征维度，2表示的是输出的特征维度，2表示的是输出的类别数，也就是输出有2个维度，每个维度表示的是一个类别的概率，这里的2表示的是二分类问题，如果是多分类问题，这里的2就要改成类别的个数，比如如果是5分类问题，这里的2就要改成5。

        if init_weights:
            initialize_weights(self)
        print('[dpcv/modeling/networks/audio_visual_residual.py] class AudioVisualResNet18Udiva - 结束执行构造函数__init__... ')

    def forward(self, aud_input, vis_input): # forward函数是必须要定义的，这个函数是用来定义网络的前向传播的，这里的aud_input和vis_input是输入的音频和视频数据, aud_input的shape是[batch_size, 1, 1, 50176], vis_input是一个list，里面的每个元素是一个tensor: [batch_size, 3, 224, 224]
        # 将音频和视频数据分别输入到音频分支和视频分支中，得到音频分支和视频分支的输出
        print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 音频和视频数据输入数据的维度为, aud_input.shape: ', aud_input.shape, ' len(vis_input)', len(vis_input), ' vis_input.shape: ', vis_input.shape) # aud_input.shape:  torch.Size([1, 2=1*2个视频, 1, 256000=16秒*采样率16000])  len(vis_input) 1  vis_input.shape:  torch.Size([1, sample_size, 6=3*2个视频, 224, 224])
        aud_x = self.audio_branch(aud_input) # aud_x的shape是[batch_size, 256, 1, 1]
        vis_x = self.visual_branch(vis_input) # vis_x的shape是[batch_size, 256, 1, 1]
        print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 音频分支和视频分支的输出数据的维度为 aud_x.shape: ', aud_x.shape, ' vis_x.shape: ', vis_x.shape)

        aud_x = aud_x.view(aud_x.size(0), -1) #view函数是用来改变tensor的形状的，这里的aud_x.size(0)表示的是batch_size, -1表示的是自动计算剩下的维度的大小，这里的aud_x的shape是[batch_size, 256]，也就是把aud_x的形状从[batch_size, 256, 1, 1]变成了[batch_size, 256]
        # vis_x = vis_x.view(vis_x.size(0), -1) # 这里的vis_x的shape是[batch_size, 256]，也就是把vis_x的形状从[batch_size, 256, 1, 1]变成了[batch_size, 256]

        print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 经过view函数之后, 音频分支和视频分支的输出数据的维度为 aud_x.shape: ', aud_x.shape, ' vis_x.shape: ', vis_x.shape)

        # 将音频分支和视频分支的输出进行cat拼接，然后输入全连接层，最后输入激活函数中得到最终输出
        feat = torch.cat([aud_x, vis_x], dim=-1) # torch.cat是用来拼接tensor的，这里的dim=-1表示的是拼接的维度，这里的feat的shape是[batch_size, 512]，也就是把aud_x和vis_x拼接在了一起
        print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 经过torch.cat函数之后即将音频分支和视频分支的输出进行拼接 拼接后得到的feat维度为 feat.shape: ', feat.shape)
        x = self.linear(feat) # self.linear = nn.Linear(512, 5) 这里的x的shape是[batch_size, 5]，也就是把feat输入到全连接层中，全连接层的输出是5个类别的概率，这里的5表示的是5个类别。全连接层的作用也就是把feat的维度从[batch_size, 512]变成了[batch_size, 5]
        print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... 将音频分支和视频分支的输出进行拼接之后的数据feat, 输入到全连接层中, 得到的输出数据x的维度为 x.shape: ', x.shape, ' x=', x)
        x = torch.sigmoid(x) # 这里的x的shape是[batch_size, 5]，也就是把x输入到sigmoid函数中，sigmoid函数的作用是把x的每一个元素都压缩到0到1之间，这里的x的每一个元素都表示的是一个类别的概率。
        print('[dpcv/modeling/networks/audio_visual_residual.py] 正在执行forward逻辑... x经过sigmoid激活函数处理后, 得到的输出数据的维度为 x.shape: ', x.shape, ' x=', x)
        # x = torch.tanh(x)
        # x = (x + 1) / 2  # scale tanh output to [0, 1]
        if self.return_feature: 
            return x, feat
        return x



class VisualResNet18(nn.Module):

    def __init__(self, init_weights=True, return_feat=False):
        super(VisualResNet18, self).__init__()
        self.return_feature = return_feat

        self.visual_branch = AudioVisualResNet(
            in_channels=3, init_stage=VisInitStage,
            block=BiModalBasicBlock, conv=[vis_conv3x3, vis_conv1x1],
            channels=[32, 64, 128, 256],
            layers=[2, 2, 2, 2]
        )
        self.linear = nn.Linear(256, 5)

        if init_weights:
            initialize_weights(self)

    def forward(self, vis_input):
        # aud_x = self.audio_branch(aud_input)
        vis_x = self.visual_branch(vis_input)

        # aud_x = aud_x.view(aud_x.size(0), -1)
        vis_x = vis_x.view(vis_x.size(0), -1)

        feat = vis_x
        x = self.linear(vis_x)
        x = torch.sigmoid(x)
        # x = torch.tanh(x)
        # x = (x + 1) / 2  # scale tanh output to [0, 1]
        if self.return_feature:
            return x, feat
        return x


class AudioResNet18(nn.Module):

    def __init__(self):
        super(AudioResNet18, self).__init__()
        self.audio_branch = AudioVisualResNet(
            in_channels=1, init_stage=AudInitStage,
            block=BiModalBasicBlock, conv=[aud_conv1x9, aud_conv1x1],
            channels=[32, 64, 128, 256],
            layers=[2, 2, 2, 2]
        )
        self.linear = nn.Linear(256, 5)

    def forward(self, aud_input):
        aud_x = self.audio_branch(aud_input)
        aud_x = aud_x.view(aud_x.size(0), -1)
        x = self.linear(aud_x)
        x = torch.sigmoid(x)
        return x


@NETWORK_REGISTRY.register()
def audiovisual_resnet(cfg=None):
    multi_modal_model = AudioVisualResNet18(return_feat=cfg.MODEL.RETURN_FEATURE) #  cfg.MODEL.RETURN_FEATURE = False
    multi_modal_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return multi_modal_model

@NETWORK_REGISTRY.register()
def audiovisual_resnet_udiva(cfg=None): # 基于UDIVA的纯CNN模型结构：没有LSTM层处理特征序列
    multi_modal_model = AudioVisualResNet18Udiva(return_feat=cfg.MODEL.RETURN_FEATURE) #  cfg.MODEL.RETURN_FEATURE = False
    multi_modal_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return multi_modal_model

@NETWORK_REGISTRY.register()
def audiovisual_resnet_lstm_udiva(cfg=None): # 基于UDIVA的CNN-LSTM模型结构：加入了LSTM层处理特征序列
    multi_modal_model = AudioVisualResNet18LSTMUdiva(return_feat=cfg.MODEL.RETURN_FEATURE) #  cfg.MODEL.RETURN_FEATURE = False
    multi_modal_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return multi_modal_model

def get_audiovisual_resnet_model():
    multi_modal_model = AudioVisualResNet18()
    multi_modal_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return multi_modal_model


@NETWORK_REGISTRY.register()
def get_audio_resnet_model(cfg=None):
    aud_modal_model = AudioResNet18()
    aud_modal_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return aud_modal_model


@NETWORK_REGISTRY.register()
def get_visual_resnet_model(cfg=None):
    visual_modal_model = VisualResNet18()
    visual_modal_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return visual_modal_model


if __name__ == "__main__":
    aud = torch.randn(2, 1, 1, 50176)
    vis = torch.randn(2, 3, 224, 224)
    # multi_model = AudioVisualResNet18()
    # y = multi_model(aud, vis)
    model = AudioResNet18()
    y = model(aud)
    print(y)

