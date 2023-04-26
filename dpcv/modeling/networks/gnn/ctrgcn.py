import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .ctrgcn_graph import Graph as CtrGraph

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: [batch_size, in_channels, T, V]
        print('[ctrgcn.py] TemporalConv input x.shape: ', x.shape) # [64, 32, 26, 20] or [64, 64, 26, 20] or [64, 64, 13, 20] or [64, 16, 52, 20] or [64, 32, 52, 20]
        x = self.conv(x)
        print('[ctrgcn.py] TemporalConv after conv x.shape: ', x.shape) # [64, 16, 52, 20] or [64, 32, 26, 20] or [64, 64, 13, 20]
        ''' 经过x = self.conv(x)后，x的维度变化：
        [64, 32, 26, 20] -> [64, 32, 26, 20] 规律: 前后不变
        [64, 64, 26, 20] -> [64, 64, 13, 20] 规律: T变为T/2
        [64, 64, 13, 20] -> [64, 64, 13, 20] 规律: 前后不变
        [64, 16, 52, 20] -> [64, 16, 52, 20] 规律: 前后不变
        
        [16, 32, 26, 20] -> [16, 32, 26, 20] 规律: 前后不变
        [16, 64, 26, 20] -> [16, 64, 13, 20] 规律: T变为T/2
        '''
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        print('[ctrgcn.py] MultiScale_TemporalConv input x.shape: ', x.shape) # [64, 64, 52, 20] or [16, 128, 26, 20]
        res = self.residual(x)
        print('[ctrgcn.py] MultiScale_TemporalConv after residual res: ', res) # 0
        branch_outs = []
        for tempconv in self.branches: # self.branches中有4个模块
            out = tempconv(x) # 经过验证，对于同样的x，每次for循环得到的out维度都是一样的
            print('[ctrgcn.py] MultiScale_TemporalConv after tempconv x.shape: ', out.shape) # [16, 16, 52, 20] or [16, 32, 26, 20] or [16, 64, 13, 20]
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        print('[ctrgcn.py] MultiScale_TemporalConv after cat out.shape: ', out.shape) # [64, 128, 26, 20] or [64, 256, 13, 20] or [64, 64, 52, 20]
        out += res
        
        ''' 根据日志总结的维度变化
        初始x: [64, 64, 52, 20] -> 输入tempconv后 x: [64, 16, 52, 20] -> cat后 out: [64, 64, 52, 20]   规律1: 初始x: (N,C,T,V) -> 输入tempconv后 x: (N,C//4,T,V) -> cat后 out: (N,C,T,V)
        初始x: [64, 256, 26, 20] -> 输入tempconv后 x: [64, 64, 13, 20] -> cat后 out: [64, 256, 13, 20] 规律2: 初始x: (N,C,T,V) -> 输入tempconv后 x: (N,C//4,T//2,V) -> cat后 out: (N,C,T//2,V)
        初始x: [64, 256, 13, 20] -> 输入tempconv后 x: [64, 64, 13, 20] -> cat后 out: [64, 256, 13, 20] 规律1: 初始x: (N,C,T,V) -> 输入tempconv后 x: (N,C//4,T,V) -> cat后 out: (N,C,T,V)
        初始x: [16, 128, 52, 20] -> 输入tempconv后 x: [16, 32, 26, 20] -> cat后 out: [16, 128, 26, 20] 规律2: 初始x: (N,C,T,V) -> 输入tempconv后 x: (N,C//4,T//2,V) -> cat后 out: (N,C,T//2,V)
        初始x: [16, 128, 26, 20] -> 输入tempconv后 x: [16, 32, 26, 20] -> cat后 out: [16, 128, 26, 20] 规律1: 初始x: (N,C,T,V) -> 输入tempconv后 x: (N,C//4,T,V) -> cat后 out: (N,C,T,V)
        初始x: [16, 256, 26, 20] -> 输入tempconv后 x: [16, 64, 13, 20] -> cat后 out: [16, 256, 13, 20] 规律2: 初始x: (N,C,T,V) -> 输入tempconv后 x: (N,C//4,T//2,V) -> cat后 out: (N,C,T//2,V)
        初始x: [16, 256, 13, 20] -> 输入tempconv后 x: [16, 64, 13, 20] -> cat后 out: [16, 256, 13, 20] 规律1: 初始x: (N,C,T,V) -> 输入tempconv后 x: (N,C//4,T,V) -> cat后 out: (N,C,T,V)
        '''
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        print('CTRGC in_channels:', in_channels, ', out_channels:', out_channels, ', rel_channels:', self.rel_channels, ', mid_channels:', self.mid_channels)
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        print('CTRGC x.shape:', x.shape, ', A.shape:', A.shape) # CTRGC x.shape: [bs, 512, sample_size, num_nodes] , A.shape: [num_nodes, num_nodes] e.g. CTRGC x.shape: [6, 512, 5, 19] , A.shape: [19, 19]
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        print('CTRGC after unsqueeze, A.shape:', A.unsqueeze(0).unsqueeze(0).shape) # CTRGC after unsqueeze, A.shape: [1, 1, num_nodes, num_nodes] e.g. [1, 1, 19, 19]
        print('CTRGC x1.shape:', x1.shape, ', x3.shape:', x3.shape) # x1:[bs, 512, num_nodes, num_nodes], x3:[bs, 512, sample_size, num_nodes] e.g. x1:[6, 512, 19, 19], x3:[6, 512, 5, 19]
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        print('CTRGC after einsum, x1.shape:', x1.shape) # [bs, 512, sample_size, num_nodes] e.g. [6, 512, 5, 19]
        return x1


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32))) 
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        print('unit_gcn A.shape:', A.shape) # [3, num_nodes, num_nodes] e.g. [3, 19, 19]
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        # {'num_class': 10, 'num_point': 20, 'num_person': 1, 'graph': 'graph.ucla.Graph', 'graph_args': {'labeling_mode': 'spatial'}}
        super(Model, self).__init__()
        # if graph is None:
        #     raise ValueError()
        # else:
        #     # Graph = import_class(graph)
        #     Graph = CtrGraph()
        #     self.graph = Graph(**graph_args)
    
        # Graph = CtrGraph()
        self.graph = CtrGraph(**graph_args)

        A = self.graph.A # 3,25,25
        print('[ctrgcn.py] A.shape: ', A.shape) # (3, 19, 19)

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # base_channel = 64
        base_channel = 512
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        # self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        # self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        # self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        # self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        # self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        # self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel, base_channel, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)

        # self.fc = nn.Linear(base_channel*4, num_class)
        self.fc = nn.Linear(base_channel, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        print('[ctrgcn.py] 模型输入参数 x shape: ', x.shape) # [16, 3, 52, 20, 1] 16 batch_size, 3 通道数, 52 帧数, 20 节点 1 未知
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size() # N: batch size, C: channel, T: frame, V: vertex, M: person
        print('[ctrgcn.py] N: ', N, 'C: ', C, 'T: ', T, 'V: ', V, 'M: ', M) # N: 16 C: 3 T: 52 V: 20 M: 1

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        print('[ctrgcn.py] after permute, x:', x.shape) # [16, 60, 52]
        x = self.data_bn(x)
        print('[ctrgcn.py] after data_bn, x:', x.shape) # [16, 60, 52]
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        print('[ctrgcn.py] after view, x:', x.shape) # [bs, nodes_features, sample_size, num_nodes] e.g. [2, 512, 5, 19]
        x = self.l1(x)
        print('[ctrgcn.py] after l1, x:', x.shape) # [16, 64, 52, 20]
        x = self.l2(x)
        print('[ctrgcn.py] after l2, x:', x.shape) # [16, 64, 52, 20]
        x = self.l3(x)
        print('[ctrgcn.py] after l3, x:', x.shape) # [16, 64, 52, 20]
        x = self.l4(x)
        print('[ctrgcn.py] after l4, x:', x.shape) # [16, 64, 52, 20]
        x = self.l5(x)
        print('[ctrgcn.py] after l5, x:', x.shape) # [16, 128, 26, 20]
        x = self.l6(x)
        print('[ctrgcn.py] after l6, x:', x.shape) # [16, 128, 26, 20]
        x = self.l7(x)
        print('[ctrgcn.py] after l7, x:', x.shape) # [16, 128, 26, 20]
        x = self.l8(x)
        print('[ctrgcn.py] after l8, x:', x.shape) # [16, 256, 13, 20]
        x = self.l9(x)
        print('[ctrgcn.py] after l9, x:', x.shape) # [16, 256, 13, 20]
        x = self.l10(x)
        print('[ctrgcn.py] after l10, x:', x.shape) # [16, 256, 13, 20]

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        print('[ctrgcn.py] after view, x:', x.shape) # [16, 1, 256, 260]
        x = x.mean(3).mean(1)
        print('[ctrgcn.py] after mean, x:', x.shape) # [16, 256]
        x = self.drop_out(x)
        print('[ctrgcn.py] after drop_out, x:', x.shape) # [16, 256]

        print('[ctrgcn.py] 模型最终返回 self.fc(x).shape:', self.fc(x).shape) # [16, 10]
        return self.fc(x)
