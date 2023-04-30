import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from scipy.spatial import distance
from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
from .resnet import resnet18, resnet50, resnet101
from .graph import create_e_matrix
from .graph_edge_model import GEM
from .basic_block import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gated GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # GNN Matrix: E x N
        # Start Matrix Item:  define the source node of one edge
        # End Matrix Item:  define the target node of one edge
        # Algorithm details in Residual Gated Graph Convnets: arXiv preprint arXiv:1711.07553
        # or Benchmarking Graph Neural Networks: arXiv preprint arXiv:2003.00982v3

        start, end = create_e_matrix(self.num_classes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U1 = nn.Linear(dim_in, dim_out, bias=False)
        self.V1 = nn.Linear(dim_in, dim_out, bias=False)
        self.A1 = nn.Linear(dim_in, dim_out, bias=False)
        self.B1 = nn.Linear(dim_in, dim_out, bias=False)
        self.E1 = nn.Linear(dim_in, dim_out, bias=False)

        self.U2 = nn.Linear(dim_in, dim_out, bias=False)
        self.V2 = nn.Linear(dim_in, dim_out, bias=False)
        self.A2 = nn.Linear(dim_in, dim_out, bias=False)
        self.B2 = nn.Linear(dim_in, dim_out, bias=False)
        self.E2 = nn.Linear(dim_in, dim_out, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)
        self.bnv1 = nn.BatchNorm1d(num_classes)
        self.bne1 = nn.BatchNorm1d(num_classes*num_classes)

        self.bnv2 = nn.BatchNorm1d(num_classes)
        self.bne2 = nn.BatchNorm1d(num_classes * num_classes)

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)

    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U1.weight.data.normal_(0, scale)
        self.V1.weight.data.normal_(0, scale)
        self.A1.weight.data.normal_(0, scale)
        self.B1.weight.data.normal_(0, scale)
        self.E1.weight.data.normal_(0, scale)

        self.U2.weight.data.normal_(0, scale)
        self.V2.weight.data.normal_(0, scale)
        self.A2.weight.data.normal_(0, scale)
        self.B2.weight.data.normal_(0, scale)
        self.E2.weight.data.normal_(0, scale)

        bn_init(self.bnv1)
        bn_init(self.bne1)
        bn_init(self.bnv2)
        bn_init(self.bne2)

    def forward(self, x, edge):
        print('[MEFL_gratis.py] GNN x.device: ', x.device, ', edge.device:', edge.device)
        # device
        dev = x.get_device()
        if dev >= 0:
            start = self.start.to(dev)
            end = self.end.to(dev)
        else:
            start = self.start
            end = self.end
        # GNN Layer 1:
        res = x
        Vix = self.A1(x)  # V x d_out
        Vjx = self.B1(x)  # V x d_out
        e = self.E1(edge)  # E x d_out
        edge = edge + self.act(self.bne1(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec',(start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b,self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V1(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U1(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = self.act(res + self.bnv1(x))
        res = x

        # GNN Layer 2:
        Vix = self.A2(x)  # V x d_out
        Vjx = self.B2(x)  # V x d_out
        e = self.E2(edge)  # E x d_out
        edge = edge + self.act(self.bne2(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V2(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U2(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = self.act(res + self.bnv2(x))
        return x, edge


class Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Head, self).__init__()
        # The head of network
        # Input: the feature maps x from backbone
        # Output: the AU recognition probabilities cl And the logits cl_edge of edge features for classification
        # Modules: 1. AFG extracts individual Au feature maps U_1 ---- U_N
        #          2. GEM: graph edge modeling for learning multi-dimensional edge features
        #          3. Gated-GCN for graph learning with node and multi-dimensional edge features
        # sc: individually calculate cosine similarity between node features and a trainable vector.
        # edge fc: for edge prediction

        self.in_channels = in_channels # 512
        self.num_classes = num_classes # 4
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels) # self.in_channels: 512
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.edge_extractor = GEM(self.in_channels, self.num_classes)
        self.gnn = GNN(self.in_channels, self.num_classes)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(7, self.in_channels))) # [7, 512]
        self.edge_fc = nn.Linear(self.in_channels, 3)
        self.fc = nn.Sequential(nn.Linear(self.num_classes*self.in_channels, 7*self.in_channels),)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.edge_fc.weight)
        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        # x: [bs, 49, 512]
        
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1)) # before layer: x=[bs, 49, 512], after layer: [bs, 1, 49, 512]
        f_u = torch.cat(f_u, dim=1) # [bs, num_classes=4, 49, 512]
        f_v = f_u.mean(dim=-2) # [bs, num_classes=4, 512], dim=-2 means the second last dim, which is 49 

        # MEFL
        # 首先创建一个形状为 (batch_size, 16, 512) 的全零张量 f_e ，其中 batch_size 表示批次大小，16 表示节点个数（这个值在实际情况中可能不同），512 表示每个节点特征的维度。
        f_e = torch.zeros(f_v.shape[0], 16, 512).cpu().detach().numpy() # [bs, 16, 512]

        # 对于每个样本，计算4个图节点两两之间的特征差异，具体做法是遍历所有节点的组合，计算每个节点特征向量的欧几里得距离，然后将结果存储在 f_e 张量的相应位置上。
        for m in range(f_v.shape[0]): # 遍历每个batch, f_v.shape: [bs, num_classes=4, 512]
          for i in range(4): # i=0,1,2,3
            for j in range(4): # j=0,1,2,3
              a = f_v[m,i,:] # 表示第 m 个样本中第 i 个节点的特征向量, e.g. f_v[0,0,:] 表示第一个batch中第一个节点的特征向量
              b = f_v[m,j,:] # 表示第 m 个样本中第 j 个节点的特征向量  e.g. f_v[0,1,:] 表示第一个batch中第二个节点的特征向量 
              f_e[m,i*j,:].fill(distance.euclidean(a.cpu().detach().numpy(), b.cpu().detach().numpy())) # 将计算得到的距离存储在 f_e 张量的相应位置上
              # e.g. i=0时，计算j为0,1,2,3时 a和b之间的euclidean距离，存储在f_e[0,0,:]中;
        # 将 f_v 和 f_e 作为输入，通过图神经网络模型 gnn 进行计算，得到更新后的节点特征 f_v 和边特征 f_e。
        f_v, f_e = self.gnn(f_v, torch.Tensor(f_e).to(device))

        b, n, c = f_v.shape # b: bs, n: num_classes=4, c: 512
        flatten_f_v = torch.flatten(f_v, start_dim=1) # [bs, num_classes=4, 512] -> [bs, 4*512]
        cl =  self.fc(flatten_f_v) # [bs, 4*512] -> [bs, 7*512] 从4变为7的原因：FER-2013数据集是7分类，参考：https://www.kaggle.com/datasets/msambare/fer2013
        cl = cl.view(b, 7, c) # [bs, 7*512] -> [bs, 7, 512]
        sc = self.sc # [7, 512]
        sc = self.relu(sc) # [7, 512]
        sc = F.normalize(sc, p=2, dim=-1) # [7, 512]
        cl = F.normalize(cl, p=2, dim=-1) # [bs, 7, 512]
        cl = (cl * sc.view(1, 7, c)).sum(dim=-1, keepdim=False) # [bs, 7, 512] * [7, 512] 
        cl_edge = self.edge_fc(f_e) # [bs, 16, 512] -> [bs, 16, 3]
        return cl, cl_edge, f_v, f_e, f_u # [bs, 7], [bs, 16, 3]


class MEFARG(nn.Module):
    def __init__(self, num_classes=7, backbone='swin_transformer_base'):
        super(MEFARG, self).__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None

        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            # self.in_channels = self.backbone.fc[0].weight.shape[1] # 2048
            self.in_channels = self.backbone.fc.weight.shape[1] # 2048
            self.out_channels = self.in_channels // 4 # 512
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels) # [bs, 49, 2048] -> [bs, 49, 512]
        self.head = Head(self.out_channels, num_classes) # self.out_channels: 512, num_classes: 4

    def forward(self, x):
        # x: b d c
        # x: [bs, 3, 224, 224]
        x = self.backbone(x) # x: [bs, 49, 2048]
        x = self.global_linear(x) # x: [bs, 49, 512]
        cl, cl_edge, f_v, f_e, f_u = self.head(x)
        return cl, cl_edge, f_v, f_e, f_u