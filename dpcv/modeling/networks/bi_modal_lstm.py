# # 为了解决ModuleNotFoundError: No module named 'dpcv'，执行以下代码
# import sys
# import os
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn
from dpcv.modeling.module.weight_init_helper import initialize_weights
from .build import NETWORK_REGISTRY
import wandb

class BiModelLSTM(nn.Module):
    def __init__(self, init_weights=True, true_personality=False):
        super(BiModelLSTM, self).__init__()
        self.audio_branch = nn.Linear(in_features=68, out_features=32)
        self.image_branch_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=9, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.image_branch_linear = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=16 * 8 * 8, out_features=1024),
            nn.Linear(in_features=1024, out_features=128),
            nn.Dropout(0.2)
        )
        # the original paper set hidden_size to 128
        # self.lstm = nn.LSTM(input_size=160, hidden_size=128)
        # self.out_linear = nn.Linear(in_features=128, out_features=5)
        self.lstm = nn.LSTM(input_size=160, hidden_size=512)
        self.out_linear = nn.Linear(in_features=512, out_features=5)
        self.true_personality = true_personality
        if init_weights:
            initialize_weights(self)

    def forward(self, audio_feature, img_feature):
        x_audio = self.audio_branch(audio_feature)  # (bs * 6, 32)
        x_img = self.image_branch_conv(img_feature)  # (bs * 6, 16 * 8 * 8)
        x_img = self.image_branch_linear(x_img)  # (bs * 6, 128)
        x = torch.cat([x_audio, x_img], dim=-1)
        x = x.view(6, -1, 160)  # x_shape = (6, bs, 160)
        x, _ = self.lstm(x)  # x_shape = (6, bs, 128)
        x = self.out_linear(x)  # x_shape = (6, bs, 5)
        x = x.permute(1, 0, 2)  # x_shape = (bs, 6, 5)
        if self.true_personality:
            return x.mean(dim=1)
        y = torch.sigmoid(x).mean(dim=1)  # y_shape = (bs, 5)
        return y


class ImgLSTM(nn.Module):
    def __init__(self):
        super(ImgLSTM, self).__init__()
        self.image_branch_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=9, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.image_branch_linear = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=16 * 8 * 8, out_features=1024),
            nn.Linear(in_features=1024, out_features=128),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128)
        self.out_linear = nn.Linear(in_features=128, out_features=5)

    def forward(self, img_feature): 
        # 问题1：这里的img_feature第一个维度是bs * 6，用6是因为一共有6帧图片，是一个图片序列。帧数为6而不是其他值是因为数据预处理时随机选取了6帧图片，具体立即可以参见：bimodal_lstm_data_loader, true_personality_lstm_visual_dataloader 等。搜索方法：看哪些config中有get_img_modal_lstm_model，然后找它对应的dataloader。
        # 问题2：为什么这里的img_feature是(bs*6, 3, 112, 112),即第一维度是一个乘积后的结果，而不是(bs, 6, 3, 112, 112)即bs和帧序列数分开来写？因为在BimodalLSTMTrainVisual中的data_fmt方法，已经执行了一个逻辑：img_in = img_in.view(-1, 3, 112, 112) ，即 初始img_in的维度是[batch_size, 6(帧序列), 3, 112, 112]，经过view后变成[batch_size*6, 3, 112, 112]
        x_img = self.image_branch_conv(img_feature)  # (bs * 6, 16 * 8 * 8)
        x_img = self.image_branch_linear(x_img)  # (bs * 6, 128)
        x = x_img.view(6, -1, 128)  # x_shape = (6, bs, 160), 6 代表图片序列有6个 view(6, -1, 128) 是指将x_img reshape成(6, bs, 128)的形状, -1表示自动计算
        x, _ = self.lstm(x)  # x_shape = (6, bs, 128)
        x = self.out_linear(x)  # x_shape = (6, bs, 5)
        x = x.permute(1, 0, 2)  # x_shape = (bs, 6, 5)
        y = torch.sigmoid(x).mean(dim=1)  # y_shape = (bs, 5)
        return y


class AudioLSTM(nn.Module):

    def __init__(self, with_sigmoid=False):
        super(AudioLSTM, self).__init__()
        self.audio_branch = nn.Linear(in_features=68, out_features=32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=128)
        self.out_linear = nn.Linear(in_features=128, out_features=5)
        self.with_sigmoid = with_sigmoid

    def forward(self, audio_feature):
        x_audio = self.audio_branch(audio_feature)  # (bs * 6, 32)
        x = x_audio.view(6, -1, 32)  # x_shape = (6, bs, 160)
        x, _ = self.lstm(x)  # x_shape = (6, bs, 128)
        x = self.out_linear(x)  # x_shape = (6, bs, 5)
        x = x.permute(1, 0, 2)  # x_shape = (bs, 6, 5)
        if self.with_sigmoid:
            y = torch.sigmoid(x).mean(dim=1)  # y_shape = (bs, 5)
        else:
            y = x.mean(dim=1)
        return y

# 用于Udiva数据集的视觉模态
class ImgLSTMUdiva(nn.Module):
    def __init__(self, sample_size=16):
        super(ImgLSTMUdiva, self).__init__()
        self.sample_size = sample_size
        self.image_branch_conv = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), # in_channels 由于udiva数据合并了一对人的image，所以是6=3通道*2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=9, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.image_branch_linear = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=16 * 22 * 22, out_features=7744), # in_features 取决于执行完 x_img = self.image_branch_conv(img_feature) 后，x_img的 shape的后三个维度的乘积
            nn.Linear(in_features=7744, out_features=128),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128)
        self.out_linear = nn.Linear(in_features=128, out_features=2)

    def forward(self, aud_input=None, img_feature=None):
        # print('[bi_modal_lstm.py] img_feature.shape', img_feature.shape) # udiva: [bs, sample_size, 6, 224, 224]
        img_feature = img_feature.view(-1, 6, 224, 224) # 将图片输入reshape成 [batch_size*sample_size, 6, 224, 224])，然后执行卷积和线性层
        
        # 手动构造一个假输入
        # bs, img_feature = 3, torch.randn((bs * 20, 6, 224, 224)) 
        
        # print('[bi_modal_lstm.py] 0-img_feature.shape', img_feature.shape) # udiva: [bs * sample_size, 6, 224, 224]
        
        x_img = self.image_branch_conv(img_feature) 
        # print('[bi_modal_lstm.py] 1-x_img.shape', x_img.shape) # udiva: x_img[bs * sample_size, 16, 22, 22]
        
        x_img = self.image_branch_linear(x_img)  
        # print('[bi_modal_lstm.py] 2-x_img.shape', x_img.shape) # udiva: [bs * sample_size, 128]
        
        x = x_img.view(self.sample_size, -1, 128)
        # print('[bi_modal_lstm.py] 3-x.shape', x.shape) # x_shape = (sample_size, bs, 128)
        
        x, _ = self.lstm(x)  
        # print('[bi_modal_lstm.py] 4-x.shape', x.shape, ' _[0].shape', _[0].shape, ' _[1].shape', _[1].shape) # x_shape = (sample_size, bs, 128) _[0].shape torch.Size([1, bs, 128])  _[1].shape torch.Size([1, bs, 128])
        
        x = self.out_linear(x)  
        # print('[bi_modal_lstm.py] 5-x.shape', x.shape) # torch.Size([sample_size, bs, 2])
        
        x = x.permute(1, 0, 2)  
        # print('[bi_modal_lstm.py] 6-x.shape', x.shape) # torch.Size([bs, sample_size, 2])
        # print('未激活前的输出x:', x) # x是得分，得分越大，表示越可能是该类别，得分越小，表示越不可能是该类别 正数表示该类别的概率越大，负数表示该类别的概率越小

        # use sigmoid as activation function
        # y = torch.sigmoid(x).mean(dim=1) 
        # wandb.config.activation='sigmoid'
        # print('[bi_modal_lstm.py] 7-y.shape', y.shape) # torch.Size([bs, 2])
        
        # use softmax as activation function
        # y_temp = torch.softmax(x, dim=2)
        # print('y_temp:', y_temp, ' y_temp.shape', y_temp.shape)
        y = torch.softmax(x, dim=2).mean(dim=1) # shape of y: [bs, 2]
        wandb.config.activation='softmax'
        # print('y:', y, ' y.shape', y.shape)
        
        '''
        print('----Sigmoid函数----------') # sigmoid 作为最后一层输出的话，那就不能吧最后一层的输出看作成一个分布了，因为加起来不为 1 https://blog.csdn.net/qq_38765642/article/details/109851437
        print('torch.sigmoid(x):', torch.sigmoid(x))
        print('torch.sigmoid(x).mean(dim=1):', y)
        
        # 测试1: 将激活函数换为softmax函数
        y_soft = torch.softmax(x, dim=2).mean(dim=1)
        
        # 测试2: 将激活函数换为relu函数
        y_relu = torch.relu(x).mean(dim=1) 
        
        # 测试3: 将激活函数换为tanh函数
        y_tanh = torch.tanh(x).mean(dim=1)
        
        print('----Softmax: 也是一种sigmoid函数------')
        print('y_soft:', y_soft)
        print('----Relu: f(x)=max(0,x)-------------')
        print('y_relu:', y_relu)
        print('----Tanh: f(x)=tanh(x)-------------')
        print('y_tanh:', y_tanh)
        '''
        
        return y

@NETWORK_REGISTRY.register()
def get_img_modal_lstm_model_udiva(cfg=None):
    img_lstm = ImgLSTMUdiva(sample_size=cfg.DATA.SAMPLE_SIZE)
    img_lstm.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return img_lstm

def get_bi_modal_lstm_model():
    bi_modal_lstm = BiModelLSTM()
    bi_modal_lstm.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return bi_modal_lstm


@NETWORK_REGISTRY.register()
def bi_modal_lstm_model(cfg=None):
    bi_modal_lstm = BiModelLSTM()
    bi_modal_lstm.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return bi_modal_lstm


@NETWORK_REGISTRY.register()
def bi_modal_lstm_model_true_personality(cfg=None):
    bi_modal_lstm = BiModelLSTM(true_personality=True)
    bi_modal_lstm.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return bi_modal_lstm


@NETWORK_REGISTRY.register()
def get_img_modal_lstm_model(cfg=None):
    img_lstm = ImgLSTM()
    img_lstm.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return img_lstm


@NETWORK_REGISTRY.register()
def get_aud_modal_lstm_model(cfg=None):
    aud_lstm = AudioLSTM()
    aud_lstm.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return aud_lstm


if __name__ == "__main__":
    """
    basically batch size is the number of video
    """
    bs = 2
    au_ft = torch.randn((bs * 6, 68))
    im_ft = torch.randn((bs * 6, 3, 112, 112))
    # bi_model = BiModelLSTM()
    # out = bi_model(au_ft, im_ft)

    # img_model = ImgLSTM()
    # out = img_model(im_ft)

    # aud_model = AudioLSTM()
    # out = aud_model(au_ft)
    # print(out.shape)
    
    udiva_im_ft = torch.randn((bs * 6, 3, 112, 112))
    print('udiva_im_ft.shape=', udiva_im_ft.shape)
    img_model = ImgLSTMUdiva()
    out = img_model(udiva_im_ft)
    print('out.shape=', out.shape)