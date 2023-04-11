import torch
import torch.nn as nn
import math


class CrossAttn(nn.Module):
    """ cross attention Module"""
    def __init__(self, in_channels):
        super(CrossAttn, self).__init__()
        self.in_channels = in_channels
        self.linear_q = nn.Linear(in_channels, in_channels // 2)
        self.linear_k = nn.Linear(in_channels, in_channels // 2)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = (self.in_channels // 2) ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.linear_k.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_q.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_v.weight.data.normal_(0, math.sqrt(2. / in_channels))

    def forward(self, y, x):
        """
        这是一个用于实现交叉注意力机制的 PyTorch 模块，主要作用是将两个输入张量 y 和 x 进行交叉注意力操作，得到一个新的输出张量 out。
        具体步骤如下：
        将输入张量 y 通过线性变换（Linear）得到查询矩阵 query；
        将输入张量 x 通过线性变换得到键矩阵 key 和值矩阵 value；
        计算查询矩阵 query 和键矩阵 key 的点积（Dot Product），并对点积进行缩放，以控制其大小范围，得到点积矩阵 dots；
        对点积矩阵 dots 沿着最后一维进行 Softmax 操作，得到注意力矩阵 attn；
        将注意力矩阵 attn 与值矩阵 value 进行点积，得到最终的输出矩阵 out。

        交叉注意力机制可以用于多个领域，如计算机视觉中的图像字幕生成、自然语言处理中的机器翻译和语音识别等任务中，可以有效地将不同领域的信息进行融合，提高模型的性能。
        """
        print('[graph_edge_model.py] CrossAttn, input y.shape: ', y.shape, ', x.shape: ', x.shape)
        query = self.linear_q(y)
        key = self.linear_k(x)
        value = self.linear_v(x)
        print('[graph_edge_model.py] CrossAttn, after linear, query.shape: ', query.shape, ', key.shape: ', key.shape, ', value.shape: ', value.shape)
        dots = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)
        print('[graph_edge_model.py] CrossAttn, after matmul, dots.shape: ', dots.shape, ', attn.shape: ', attn.shape)
        out = torch.matmul(attn, value)
        print('[graph_edge_model.py] CrossAttn, after matmul, out.shape: ', out.shape)
        return out


class GEM(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GEM, self).__init__()
        print('[graph_edge_model.py] GEM, in_channels: ', in_channels, ', num_classes: ', num_classes)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.FAM = CrossAttn(self.in_channels)
        self.ARM = CrossAttn(self.in_channels)
        self.edge_proj = nn.Linear(in_channels, in_channels)
        self.bn = nn.BatchNorm2d(self.num_classes * self.num_classes)

        self.edge_proj.weight.data.normal_(0, math.sqrt(2. / in_channels))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, class_feature, global_feature):
        print('[graph_edge_model.py] class_feature.shape: ', class_feature.shape, ', global_feature.shape: ', global_feature.shape) # class_feature.shape: [bs, 12, 49, 512] , global_feature.shape: [bs, 49, 512]
        B, N, D, C = class_feature.shape # B: bs, N: 12, D: 49, C: 512
        if self.num_classes != 19: # 12 or 8 or 4
            global_feature = global_feature.repeat(1, N, 1).view(B, N, D, C)
        print('[graph_edge_model.py] before FAM, class_feature.shape: ', class_feature.shape, ', global_feature.shape: ', global_feature.shape) # class_feature.shape: [bs, 12, 49, 512] , global_feature.shape: [bs, 12, 49, 512]
        feat = self.FAM(class_feature, global_feature) # class_feature: [bs, 12, 49, 512], global_feature: [bs, 12, 49, 512]
        print('[graph_edge_model.py] after FAM, feat.shape: ', feat.shape) # feat.shape: [bs, 12, 49, 512]
        feat_end = feat.repeat(1, 1, N, 1).view(B, -1, D, C) # [bs, 12, 49, 512] -> repeat(1, 1, 12, 1) -> [bs, 144, 49, 512]
        feat_start = feat.repeat(1, N, 1, 1).view(B, -1, D, C)
        feat = self.ARM(feat_start, feat_end) # feat_start: [bs, 144, 49, 512], feat_end: [bs, 144, 49, 512]
        print('[graph_edge_model.py] after ARM, feat.shape: ', feat.shape, ', input param is feat_start.shape:', feat_start.shape, ', feat_end.shape:', feat_end.shape) # feat.shape: [bs, 144, 49, 512], input param is feat_start.shape: [bs, 144, 49, 512], feat_end.shape: [bs, 144, 49, 512]
        edge = self.bn(self.edge_proj(feat))
        print('[graph_edge_model.py] after edge_proj, final edge.shape: ', edge.shape) # edge.shape: [bs, 144, 49, 512]
        return edge



