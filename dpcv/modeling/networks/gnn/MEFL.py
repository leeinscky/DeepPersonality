import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
from .resnet import resnet18, resnet50, resnet101
from .resnet_3d import resnet50_3d
from .graph import create_e_matrix
from .graph_edge_model import GEM
from .basic_block import *
from dpcv.modeling.module.weight_init_helper import initialize_weights
from dpcv.modeling.networks.build import NETWORK_REGISTRY
# from .MEFL_gratis import MEFARG as MEFARG_GRATIS
# from .ctrgcn import Model as CTRGCN
from .stsgcn import STSGCN
import gc
import time
from multiprocessing import Pool
import torch.multiprocessing as mp
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_half = False
AU_12CLASS = 12
AU_8CLASS = 8
AU_15CLASS = 15


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
        # print('[MEFL.py] GNN, input x.shape:', x.shape, ', edge.shape:', edge.shape) # x.shape: [bs, 12, 512], edge.shape: [bs, 144, 512] 因为BP4D有12个节点，144个边
        # device
        dev = x.get_device()
        # print('[MEFL.py] GNN, device: ', dev) # cpu: -1, gpu: 0, 1, 2, ...
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
        if torch.cuda.is_available() and use_half:
            start = start.half()
            end = end.half()
        # print('[MEFL.py] GNN, Vix.dtype:', Vix.dtype, ', Vjx.dtype:', Vjx.dtype, ', e.dtype:', e.dtype, ', start.dtype:', start.dtype, ', end.dtype:', end.dtype)
        # Vix.dtype: torch.float16 , Vjx.dtype: torch.float16 , e.dtype: torch.float16 , start.dtype: torch.float32 , end.dtype: torch.float32
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
        ## print('[MEFL.py] GNN, output x.shape:', x.shape, ', edge.shape:', edge.shape)
        return x, edge


class Head(nn.Module):
    def __init__(self, in_channels, num_classes, modal):
        super(Head, self).__init__()
        # The head of network
        # Input: the feature maps x from backbone
        # Output: the AU recognition probabilities cl And the logits cl_edge of edge features for classification
        # Modules: 1. AFG extracts individual Au feature maps U_1 ---- U_N
        #          2. GEM: graph edge modeling for learning multi-dimensional edge features
        #          3. Gated-GCN for graph learning with node and multi-dimensional edge features
        # sc: individually calculate cosine similarity between node features and a trainable vector.
        # edge fc: for edge prediction
        self.modal = modal # visual or audio
        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.edge_extractor = GEM(self.in_channels, self.num_classes)
        self.gnn = GNN(self.in_channels, self.num_classes)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.edge_fc = nn.Linear(self.in_channels, 1)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.edge_fc.weight)
        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        # print('\n[MEFL.py] class Head.forward, input x.shape: ', x.shape) # x.shape: [bs, 49, 512]
        
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)
        # print('[MEFL.py] class Head.forward, f_u.shape: ', f_u.shape, 'f_v.shape: ', f_v.shape) # f_u.shape: [bs, 12, 49, 512], f_v.shape: [bs, 12, 512]

        # MEFL
        f_e = self.edge_extractor(f_u, x) # GEM: graph edge modeling for learning multi-dimensional edge features
        # print('[MEFL.py] class Head.forward, edge_extractor 的输入参数 f_u.shape: ', f_u.shape, 'x.shape: ', x.shape, ', 计算结果 f_e.shape: ', f_e.shape) 
        # f_u.shape: [bs, 12, 49, 512], x.shape: [bs, 49, 512], 计算结果 f_e.shape: [bs, 144, 49, 512]
        f_e = f_e.mean(dim=-2)
        
        # print('[MEFL.py] class Head.forward, GNN的输入参数, f_v.shape: ', f_v.shape, 'f_e.shape: ', f_e.shape) # f_v.shape: [bs, 12, 512], f_e.shape: [bs, 144, 512]
        f_v, f_e = self.gnn(f_v, f_e) # Gated-GCN for graph learning with node and multi-dimensional edge features
        # print('[MEFL.py] class Head.forward, after self.gnn, f_v.shape: ', f_v.shape, 'f_e.shape: ', f_e.shape) # f_v.shape: [bs, 12, 512], f_e.shape: [bs, 144, 512]

        b, n, c = f_v.shape
        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1) # cl means class logits
        # print('[MEFL.py] class Head.forward, after F.normalize, cl.shape: ', cl.shape, 'sc.shape: ', sc.shape) # cl.shape: [bs, 12, 512], sc.shape: [12, 512]
        
        cl = (cl * sc.view(1, n, c)).sum(dim=-1, keepdim=False)
        cl_edge = self.edge_fc(f_e)
        # print('[MEFL.py] class Head.forward, final return cl.shape: ', cl.shape, 'cl_edge.shape: ', cl_edge.shape) # cl.shape: [bs, 12], cl_edge.shape: [bs, 144, 4]
        # return cl, cl_edge, f_v, f_e, f_u
        if self.modal == 'visual':
            return f_v, f_u
        elif self.modal == 'visual_simple':
            return f_v, f_e, cl, cl_edge
        elif self.modal == 'audio':
            return f_v, f_u, cl, cl_edge
        else:
            raise ValueError('modal should be visual or audio')


class MEFARG(nn.Module):
    def __init__(self, num_classes=12, backbone='swin_transformer_base', modal='visual', cfg=None):
        super(MEFARG, self).__init__()
        # print('[MEFL.py] class MEFARG.__init__ backbone: ', backbone)
        self.modal = modal
        if self.modal == 'visual':
            self.channels = 3
            self.use_pretrained_backbone = True
        elif self.modal == 'visual_simple':
            self.channels = 6
            self.use_pretrained_backbone = False
        elif self.modal == 'audio':
            self.channels = 2
            self.use_pretrained_backbone = False
        else:
            raise ValueError('modal should be visual or audio or visual_simple')
        self.cfg = cfg
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
            elif backbone == 'resnet50_3d':
                self.backbone = resnet50_3d(pretrained=self.use_pretrained_backbone, in_channels=self.channels, n_classes=self.cfg.MODEL.NUM_CLASS)
            else:
                # self.backbone = resnet50() # pretrained=True
                self.backbone = resnet50(pretrained=self.use_pretrained_backbone, in_channels=self.channels) # pretrained=False
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        # print('[MEFL.py] class MEFARG.__init__ self.in_channels:', self.in_channels, 'self.out_channels:', self.out_channels)
        # MEFARG_resnet101_BP4D_fold1.pth: 2048, 512; MEFARG_resnet50_BP4D_fold1.pth: 2048, 512;
        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, num_classes, self.modal)

    def forward(self, x):
        # x: b d c
        # x = torch.randn(2, 3, 112, 112) # or x = torch.randn(2, 3, 224, 224) # temp test
        # [bs, sample_size, 6(3*2person=6), 112, 112] -> [bs, 6(3*2person=6), sample_size, 112, 112]
        if self.modal == 'visual_simple':
            x = x.permute(0, 2, 1, 3, 4) # .contiguous()
        print('[MEFL.py] MEFARG.forward, input x.shape: ', x.shape) # 3 channels: x.shape=[bs, 3, 112, 112]; 6 channels: x.shape=[bs, 6, 112, 112]
        x = self.backbone(x)
        print('[MEFL.py] MEFARG.forward, after backbone, x.shape: ', x.shape) # 3 channels: x.shape=[bs, 16, 2048]; 6 channels: x.shape=[bs, 16, 2048]
        x = self.global_linear(x)
        # print('[MEFL.py] MEFARG.forward, after global_linear, x.shape: ', x.shape) # 3 channels: x.shape=[bs, 16, 512]; 6 channels: x.shape=[bs, 16, 512]
        # cl, cl_edge, f_v, f_e, f_u = self.head(x)
        if self.modal == 'visual':
            f_v, f_u = self.head(x)
            return f_v, f_u
        elif self.modal == 'visual_simple':
            f_v, f_e, cl, cl_edge = self.head(x)
            print('[MEFL.py] MEFARG.forward, final f_v.shape: ', f_v.shape, 'f_e.shape: ', f_e.shape)
            print('[MEFL.py] MEFARG.forward, after head, cl.shape: ', cl.shape, 'cl_edge.shape: ', cl_edge.shape)
            return f_v, f_e, cl, cl_edge
        elif self.modal == 'audio':
            f_v, f_e, cl, cl_edge = self.head(x)
            print('[MEFL.py] MEFARG.forward, final f_v.shape: ', f_v.shape, 'f_e.shape: ', f_e.shape)
            print('[MEFL.py] MEFARG.forward, after head, cl.shape: ', cl.shape, 'cl_edge.shape: ', cl_edge.shape) # cl.shape: [bs, num_class], cl_edge.shape: [bs, num_class*num_class, num_class]
            return f_v, f_e, cl, cl_edge
        else:
            raise ValueError('modal should be visual or audio')


def load_state_dict(model,path):
    checkpoints = torch.load(path,map_location=device)
    state_dict = checkpoints['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict,strict=False)
    return model


def union_au_class(au_12class, au_8class, type='v'):
    """
    if type == 'v':
        au_12class: [bs, 12, 512]
        au_8class: [bs, 8, 512]
        return: [bs, 15, 512]
    elif type == 'u':
        au_12class: [bs, 12, 49, 512]
        au_8class: [bs, 8, 49, 512]
        return: [bs, 15, 49, 512]
    """
    bs = au_12class.shape[0]
    
    if type == 'v':
        au_15class = torch.zeros(bs, 15, au_12class.shape[2])
        '''
        # original logic
        # au_12class的第二个维度是12，这12个分别对应于: AU1, AU2, AU4, AU6, AU7, AU10, AU12, AU14, AU15, AU17, AU23, AU24
        # au_8class的第二个维度是8，这8个分别对应于: AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26
        # au_12class 和 au_8class的并集是: AU1, AU2, AU4, AU6, AU7, AU9, AU10, AU12, AU14, AU15, AU17, AU23, AU24, AU25, AU26 共15个AU
        # au_8class 和 au_12class的不同AU是: AU9, AU25, AU26
        # au_15class[:, 0] = au_12class[:, 0] # AU1
        # au_15class[:, 1] = au_12class[:, 1] # AU2
        # au_15class[:, 2] = au_12class[:, 2] # AU4
        # au_15class[:, 3] = au_12class[:, 3] # AU6
        # au_15class[:, 4] = au_12class[:, 4] # AU7
        # au_15class[:, 5] = au_8class[:, 5] # AU9
        # au_15class[:, 6] = au_12class[:, 5] # AU10
        # au_15class[:, 7] = au_12class[:, 6] # AU12
        # au_15class[:, 8] = au_12class[:, 7] # AU14
        # au_15class[:, 9] = au_12class[:, 8] # AU15
        # au_15class[:, 10] = au_12class[:, 9] # AU17
        # au_15class[:, 11] = au_12class[:, 10] # AU23
        # au_15class[:, 12] = au_12class[:, 11] # AU24
        # au_15class[:, 13] = au_8class[:, 6] # AU25
        # au_15class[:, 14] = au_8class[:, 7] # AU26
        '''
        # simplified
        au_15class[:, :5, :] = au_12class[:, :5, :] # AU1-AU7         au_15class的第1到第5列
        au_15class[:, 5, :] = au_8class[:, 5, :] # AU9                au_15class的第6列
        au_15class[:, 6:13, :] = au_12class[:, 5:12, :] # AU10-AU24   au_15class的第7到第13列 au_15class[:, 6:13] 是指 au_15class 在第二维中从索引 6 开始（包括第 6 列），到索引 13 结束（不包括第 13 列）的子数组。换句话说，它包含 au_15class 中的第 7-13 列。
        au_15class[:, 13:15, :] = au_8class[:, 6:8, :] # AU25-AU26    au_15class的第14到第15列
    elif type == 'u':
        au_15class = torch.zeros(bs, 15, au_12class.shape[2], au_12class.shape[3])
        au_15class[:, :5, :, :] = au_12class[:, :5, :, :] # AU1-AU7
        au_15class[:, 5, :, :] = au_8class[:, 5, :, :] # AU9
        au_15class[:, 6:13, :, :] = au_12class[:, 5:12, :, :] # AU10-AU24
        au_15class[:, 13:15, :, :] = au_8class[:, 6:8, :, :] # AU25-AU26
        
    # print('au_12class: ', au_12class, '\nau_8class: ', au_8class, '\nau_15class: ', au_15class)
    del au_12class, au_8class
    return au_15class


def process_node(au_15class, au_4class, type='v'):
    # print('[MEFL.py] process_node, input au_15class.shape: ', au_15class.shape, 'au_4class.shape: ', au_4class.shape) # au_15class.shape: [bs, 15, 512] au_4class.shape: [bs, 4, 512]
    if type == 'v':
        # [bs, 15, 512], [bs, 4, 512] -> [bs, 19, 512]
        au_19class = torch.cat((au_15class, au_4class), dim=1)
    elif type == 'u':
        # [bs, 15, 49, 512], [bs, 4, 49, 512] -> [bs, 19, 512]
        au_19class = torch.cat((au_15class, au_4class), dim=1)
    return au_19class


def construct_adj(A, steps):
    """
    构建local 时空图
    :param A: np.ndarray, adjacency matrix, shape is (N, N)
    :param steps: 选择几个时间步来构建图
    :return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    """
    
    N = len(A)  # 获得节点数
    adj = np.zeros((N * steps, N * steps, A.shape[2]))

    for i in range(steps):
        """对角线代表各个时间步自己的空间图，也就是A"""
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N, :] = A
    # print('[MEFL.py] construct_adj, adj.shape: ', adj.shape)

    # 关于下列这个循环的issue： https://github.com/SmallNana/STSGCN_Pytorch/issues/7  3个时间步的子图的确是相同的，但是卷积是基于“大图”进行的，随着卷积层的堆叠，顶层的输出融入也就融入了相邻时间的信息。
    for i in range(N):
        for k in range(steps - 1):
            """每个节点只会连接相邻时间步的自己"""
            adj[k * N + i, (k + 1) * N + i, :] = 1
            adj[(k + 1) * N + i, k * N + i, :] = 1
    for i in range(len(adj)):
        """加入自回"""
        adj[i, i] = 1
    # print('[MEFL.py] construct_adj, final adj.shape:', adj.shape)
    return adj


class GraphModel(nn.Module):
    def __init__(self, init_weights=True, num_class=2, pretrained_model=None, backbone='resnet50', sample_size=16, cfg=None):
        super(GraphModel, self).__init__()
        self.cfg = cfg
        if init_weights:
            initialize_weights(self)
        self.edge_extractor = GEM(512, 15).to(device)
        
        ### one graph model
        # self.graph_model = MEFARG(num_classes=num_class, backbone=backbone)
        # if pretrained_model is not None:
        #     self.graph_model = load_state_dict(self.graph_model, pretrained_model)
        
        ### two graph model, 12 class + 8 class, then obtain the union: a totoal of 15 AU class 取并集获得15个AU, refer to the paper: https://arxiv.org/pdf/2205.01782.pdf
        '''
        1. swin_transformer: backbone="swin_transformer_tiny" 和 "swin_transformer_small" 时，输出的特征维度都是384；当backbone="swin_transformer_base"时，输出的特征维度是512
        2. resnet: backbone="resnet50" 
        '''
        self.graph_model_12class = MEFARG(num_classes=12, backbone="resnet50", modal='visual').to(device)
        self.graph_model_8class = MEFARG(num_classes=8, backbone="resnet50", modal='visual').to(device)
        ### graph model for 4 nodes features, refer to the paper: https://arxiv.org/pdf/2211.12482.pdf
        # self.graph_model_4class = MEFARG_GRATIS(num_classes=4, backbone="resnet50").to(device)
        # print('[MEFL.py] self.graph_model_12class device:', next(self.graph_model_12class.parameters()).device, 'self.graph_model_8class device: ', next(self.graph_model_8class.parameters()).device)
        if pretrained_model is not None:
            print('[MEFL.py] MEFARG load pretrained model: ', pretrained_model)
            self.graph_model_12class = load_state_dict(self.graph_model_12class, pretrained_model[0])
            self.graph_model_8class = load_state_dict(self.graph_model_8class, pretrained_model[1])
        
        self.sts_gcn = STSGCN(
            # adj=torch.randn(15, 15).to(device),
            adj=torch.randn(1, 1, 1).to(device), # TODO
            history=sample_size,
            num_of_vertices=15*2, # two person, 15 for each person
            # num_of_vertices=15,
            in_dim=512,
            hidden_dims=[[64, 64, 64], [64, 64, 64], [64, 64, 64], [64, 64, 64]],
            # hidden_dims=[[64, 64, 64]],
            first_layer_embedding_size=64,
            out_layer_dim=128,
            activation="GLU",
            use_mask=True,
            temporal_emb=True,
            spatial_emb=True,
            horizon=12,
            strides=3
        ).to(device)
        
        # NoXi: num_class=4; Udiva: num_class=1, num_point = union(12, 8) + 4
        # self.ctr_gcn = CTRGCN(num_class=4, num_point=19, num_person=1, graph_args={'labeling_mode': 'spatial'}, in_channels=512)
        
        self.linear_edge = nn.Linear(450, 900) # num_nodes * num_nodes * 2 = 15 * 15 * 2 = 450; (num_nodes * 2) * (num_nodes * 2) = 30 * 30 = 900
        self.linear = nn.Linear(30, 4)
        # self.linear_edge = nn.Linear(225, 900)
        # self.linear = nn.Linear(15, 4)
        
        self.node_features, self.edge_features = [], []

    def process_face(self, face_image):
        # print('[process_face] face_image: ', face_image.shape)
        # cl1_12class, cl_edge1_12class, f_v_12class, f_e_12class, f_u_12class = self.graph_model_12class(face_image)
        # cl1_8class, cl_edge1_8class, f_v_8class, f_e_8class, f_u_8class = self.graph_model_8class(face_image)
        # print('[MEFL.py] process_face, cl1_12class.shape: ', cl1_12class.shape, ', cl1_8class.shape: ', cl1_8class.shape, ', cl_edge1_12class.shape: ', cl_edge1_12class.shape, ', cl_edge1_8class.shape: ', cl_edge1_8class.shape) # cl1_12class.shape:[bs, 12], cl1_8class.shape:[bs, 8], cl_edge1_12class.shape:[bs, 144, 4], cl_edge1_8class.shape:[bs, 64, 4]
    
        f_v_12class, f_u_12class = self.graph_model_12class(face_image)
        f_v_8class, f_u_8class = self.graph_model_8class(face_image)
        # print('[MEFL.py] process_face, f_v_12class.shape: ', f_v_12class.shape, ', f_v_8class.shape: ', f_v_8class.shape, ', f_u_12class.shape: ', f_u_12class.shape, ', f_u_8class.shape: ', f_u_8class.shape) #  f_v_12class.shape:  torch.Size([1, 12, 512]) , f_v_8class.shape:  torch.Size([1, 8, 512]) , f_u_12class.shape:  torch.Size([1, 12, 49, 512]) , f_u_8class.shape:  torch.Size([1, 8, 49, 512])
        # print('[MEFL.py] process_face, f_v_12class.device:', f_v_12class.device, ', f_u_12class.device: ', f_u_12class.device, ', f_u_8class.device: ', f_u_8class.device)
        f_v_15class = union_au_class(f_v_12class.to(device), f_v_8class.to(device), type='v').to(device)
        f_u_15class = union_au_class(f_u_12class.to(device), f_u_8class.to(device), type='u').to(device)
        # print('[MEFL.py] process_face, f_v_15class.shape: ', f_v_15class.shape, ', f_u_15class.shape: ', f_u_15class.shape) # f_v_15class: [bs, 15, 512], f_u_15class: [bs, 15, 49, 512]
        
        '''
        # cl1_4class, cl_edge1_4class, f_v_4class, f_e_4class, f_u_4class = self.graph_model_4class(face_image)
        # print('[MEFL.py] process_face, f_v_4class.shape: ', f_v_4class.shape, ', f_e_4class.shape: ', f_e_4class.shape) # f_v_4class:[bs, 4, 512] f_e_4class: [bs, 16, 512]
        
        ### process node features 15 class + 4 class, and get the final node features 19 class
        # f_v_19class = process_node(f_v_15class.to(device), f_v_4class.to(device), type='v').to(device)
        # f_u_19class = process_node(f_u_15class.to(device), f_u_4class.to(device), type='u').to(device)
        # print('[MEFL.py] process_face, f_v_19class.shape: ', f_v_19class.shape, ', f_u_19class: ', f_u_19class.shape) # f_v_19class:[bs, 19, 512], f_u_19class:[bs, 19, 49, 512]
        
        # graph edge modeling for learning multi-dimensional edge features of 19 class
        f_e_19class = self.edge_extractor(f_u_19class, f_u_19class)
        print('[MEFL.py] process_face, f_e_19class.shape: ', f_e_19class.shape) # [bs, 361, 49, 512]
        f_e_19class = torch.mean(f_e_19class, dim=2) # [bs, 361, 49, 512] -> [bs, 361, 512]
        '''
        # graph edge modeling for learning multi-dimensional edge features of 15 class
        f_e_15class = self.edge_extractor(f_u_15class, f_u_15class)
        # print('[MEFL.py] process_face, f_e_15class.shape: ', f_e_15class.shape) # [bs, 225, 49, 512]
        f_e_15class = torch.mean(f_e_15class, dim=2) # [bs, 225, 49, 512] -> [bs, 225, 512]

        del f_v_12class, f_v_8class, f_u_12class, f_u_8class, f_u_15class
        if torch.cuda.is_available():
            time_start = time.time()
            gc.collect()
            torch.cuda.empty_cache()
            duration = round(time.time() - time_start, 3)
            if duration > 1:
                print('[MEFL.py] process_face, time cost of gc.collect: ', duration, 's')
        return f_v_15class, f_e_15class

    """
    def generate_frame_graph(self, one_sample_feature, sample_id, node_features, edge_features):
        start_time = time.time()
        # print('[generate_frame_graph], one_sample_feature.shape: ', one_sample_feature.shape, ', sample_id: ', sample_id)
        # one_sample_feature = x[:, i, :, :, :] # 取sample_size个中的一个sample的人脸特征: [bs, c, w, h]
        person_face_1, person_face_2 = one_sample_feature[:, 0:3, :, :], one_sample_feature[:, 3:6, :, :]
        # del one_sample_feature
        # print('[MEFL.py] GraphModel.forward, person_face_1.device: ', person_face_1.device, ', person_face_2.device: ', person_face_2.device)
        
        # concat(person_face_1: [bs, 3, 112, 112], person_face_2: [bs, 3, 112, 112]) -> [bs, 3, 224, 224]
        two_person_face = torch.cat((person_face_1, person_face_2), dim=2)
        # print('[MEFL.py] GraphModel.forward, two_person_face.shape: ', two_person_face.shape) # [bs, 3, 224, 112]
        
        # f_v_15class_1, f_e_15class_1 = self.process_face(person_face_1) # person 1 f_v_15class_1:[bs, 19, 512] f_e_15class_1:[bs, 361, 512]
        # f_v_15class_2, f_e_15class_2 = self.process_face(person_face_2) # person 2 f_v_15class_2:[bs, 19, 512] f_e_15class_2:[bs, 361, 512]
        del person_face_1, person_face_2
        # print('[MEFL.py] GraphModel.forward, f_v_15class_1.shape: ', f_v_15class_1.shape, ', f_e_15class_1.shape: ', f_e_15class_1.shape)
        # print('[MEFL.py] GraphModel.forward, f_v_15class_1.shape: ', f_v_15class_1.shape, ', f_e_15class_1.shape: ', f_e_15class_1.shape, ', f_v_15class_2.shape: ', f_v_15class_2.shape, ', f_e_15class_2.shape: ', f_e_15class_2.shape)
        
        # f_v_15class = torch.cat((f_v_15class_1, f_v_15class_2), dim=1) # [bs, n_nodes(15), 512] -> [bs, n_nodes*2(30), 512] two person
        # f_e_15class = torch.cat((f_e_15class_1, f_e_15class_2), dim=1) # [bs, 361, 512] -> [bs, 722, 512] two person # TODO
        # print('two_person_face.device: ', two_person_face.device) # if use cuda, then two_person_face.device:  cuda:0
        f_v_15class, f_e_15class = self.process_face(two_person_face)
        # print('[MEFL.py] GraphModel.forward, f_v_15class.shape: ', f_v_15class.shape, ', f_e_15class.shape: ', f_e_15class.shape) # [bs, 15, 512], [bs, 225, 512]
        # print('[MEFL.py] GraphModel.forward, f_v_15class.device: ', f_v_15class.device, ', f_e_15class.device: ', f_e_15class.device) # if use cuda, then f_v_15class.device:  cuda:0 , f_e_15class.device:  cuda:0
        '''
        # cl2, cl_edge2 = self.graph_model(person_face_2)
        # print('[MEFL.py] GraphModel.forward, cl1.shape: ', cl1.shape, ', cl_edge1.shape: ', cl_edge1.shape, ', cl2.shape: ', cl2.shape, ', cl_edge2.shape: ', cl_edge2.shape)
        
        # cl = torch.cat((cl1, cl2), dim=1)
        # cl_edge = torch.cat((cl_edge1, cl_edge2), dim=1)
        # print('[MEFL.py] GraphModel.forward, cl.shape: ', cl.shape, ', cl_edge.shape: ', cl_edge.shape)
        '''
        # print('[generate_frame_graph], 1-self.node_features:', self.node_features, ', \nself.edge_features:', self.edge_features)
        # print('[generate_frame_graph], 1-node_features:', node_features, ', \nedge_features:', edge_features)
        self.node_features[sample_id] = f_v_15class.detach()
        self.edge_features[sample_id] = f_e_15class.detach()
        # node_features[sample_id] = f_v_15class.detach()
        # edge_features[sample_id] = f_e_15class.detach()
        # print('[generate_frame_graph], 2-self.node_features:', self.node_features)
        # print('[generate_frame_graph], 2-node_features:', node_features, ', \nedge_features:', edge_features)
        
        # del f_v_15class, f_e_15class, f_v_15class_1, f_e_15class_1, f_v_15class_2, f_e_15class_2
        # del f_v_15class, f_e_15class
        
        if torch.cuda.is_available():
            time_start = time.time()
            gc.collect()
            # torch.cuda.empty_cache()
            duration = round(time.time() - time_start, 3)
            if duration > 1:
                print('[MEFL.py] GraphModel.generate_graph, current sample:', sample_id, ', gc.collect() time cost: ', duration, 's')
        
        print('[generate_frame_graph] sample_id:', sample_id, ', time cost:', round(time.time() - start_time, 3), 's')
        # return (f_v_15class.detach().cpu(), f_e_15class.detach().cpu(), sample_id)
        # return (f_v_15class, f_e_15class, sample_id)
        pass
    """

    """
    def generate_graph(self, x):
        """" 使用 torch.multiprocessing 生成图，但是耗时很久而且会报错，暂时不用""""
        print('start generate_graph')
        start_time = time.time()
        mp.set_start_method('spawn', force=True)
        sample_size = x.shape[1]
        empty_tensor = torch.empty((0)).to(device)
        print('[generate_graph] mp, after set_start_method, time cost:', round(time.time() - start_time, 3), 's') # GPU: 0.0 s
        # self.node_features = mp.Manager().list([empty_tensor.clone() for i in range(sample_size)]) # 使用共享内存
        # self.edge_features = mp.Manager().list([empty_tensor.clone() for i in range(sample_size)])
        manager = mp.Manager()
        self.node_features = manager.list([empty_tensor for i in range(sample_size)]) # 使用共享内存
        self.edge_features = manager.list([empty_tensor for i in range(sample_size)])
        print('[generate_graph] mp, after mp.Manager, time cost:', round(time.time() - start_time, 3), 's') # GPU: 5.194 s
        # print('[MEFL.py] GraphModel.generate_graph, node_features.deivce:', node_features.device, ', edge_features.device:', edge_features.device)
        # print('[generate_graph] 1-self.node_features:', self.node_features, ', self.edge_features:', self.edge_features, len(self.node_features), len(self.edge_features)) # self.node_features: [None, None, None, None, None, None, None, None, None, None] , self.edge_features: [None, None, None, None, None, None, None, None, None, None] 10 10

        print('os.cpu_count():', os.cpu_count())
        print('[generate_graph] mp, before for loop, time cost:', round(time.time() - start_time, 3), 's') # GPU: 5.194 s
        processes = []
        for sample_id in range(sample_size):
            # print('process sample:', sample_id, '...')
            # time.sleep(0.1)
            p = mp.Process(target=self.generate_frame_graph, args=(x[:, sample_id, :, :, :], sample_id, self.node_features, self.edge_features))
            p.start()
            processes.append(p)
            print('[generate_graph] mp, after sample_id:', sample_id, ', time cost:', round(time.time() - start_time, 3), 's') 
            # after sample_id: 0 , time cost: 15.849 s; 
            # after sample_id: 1 , time cost: 25.762 s; 
            # after sample_id: 2 , time cost: 35.924 s; 
            # after sample_id: 3 , time cost: 45.661 s ; 
            # after sample_id: 4 , time cost: 55.93 s; .... after sample_id: 9 , time cost: 107.313 s
        print('[generate_graph] mp, after for loop, time cost:', round(time.time() - start_time, 3), 's') # GPU: 107.313 s
        for p in processes:
            p.join()
        print('[generate_graph] mp, after join loop, time cost:', round(time.time() - start_time, 3), 's') # GPU: 111.365 s
        
        # get_ret_start = time.time()
        # for ret in results:
        #     f_v_15class, f_e_15class, sample_id = ret.get()
        #     self.node_features[sample_id] = f_v_15class
        #     self.edge_features[sample_id] = f_e_15class
        # print('[generate_graph] get_ret time cost:', round(time.time() - get_ret_start, 3), 's')
        # print('[generate_graph] All subprocesses done. time cost:', round(time.time() - start_time, 3), 's')
        
        # print('[generate_graph] self.node_features:', self.node_features, ', self.edge_features:', self.edge_features)
        # return self.node_features, self.edge_features
        # return tuple(self.node_features), tuple(self.edge_features)
        # print('[generate_graph] node_features:', node_features, ', edge_features:', edge_features)
        # return node_features, edge_features
        # return tuple(node_features).clone(), tuple(edge_features).clone()
        # return tuple(node_features).clone().cpu(), tuple(edge_features).clone().cpu()
        pass
    """

    """
    def generate_graph_cpu(self, x):
        """" 使用 python自带multiprocessing 生成图，但是无法用于GPU上，只适用于CPU，暂时不用""""
        start_time = time.time()
        sample_size = x.shape[1]
        # self.node_features 初始化为含有sample_size个空张量tensor的列表
        # empty_tensor = torch.empty((sample_size, 15, 512))
        empty_tensor = torch.empty((0, 1))
        self.node_features = [empty_tensor.clone() for i in range(sample_size)]
        self.edge_features = [empty_tensor.clone() for i in range(sample_size)]
        # print('[generate_graph] 1-self.node_features:', self.node_features, ', self.edge_features:', self.edge_features, len(self.node_features), len(self.edge_features)) # self.node_features: [None, None, None, None, None, None, None, None, None, None] , self.edge_features: [None, None, None, None, None, None, None, None, None, None] 10 10
        print('os.cpu_count():', os.cpu_count())
        results = []
        p = Pool(sample_size)
        print('[generate_graph] after create Pool, time cost:', round(time.time() - start_time, 3), 's')
        for sample_id in range(sample_size):
            # print('process sample:', sample_id, '...')
            # time.sleep(0.1)
            ret = p.apply_async(self.generate_frame_graph, args=(x[:, sample_id, :, :, :].detach(), sample_id, self.node_features, self.edge_features))
            results.append(ret)
            # ret = p.apply_async(self.generate_frame_graph, args=(x[:, sample_id, :, :, :].detach(), sample_id))
            # print('result:', result, ', type(result):', type(result), ', result.get():', result.get(), ', type(result.get()):', type(result.get()))
            # print('[generate_graph] ret:', ret, ', type(ret):', type(ret), ', type(ret.get()):', type(ret.get())) # ret: <multiprocessing.pool.ApplyResult object at 0x15016ca050a0>, type(ret): <class 'multiprocessing.pool.ApplyResult'>, type(ret.get()): <class 'tuple'>
            # f_v_15class, f_e_15class = ret.get()
            # print('[generate_graph] f_v_15class:', f_v_15class.shape, ', f_e_15class:', f_e_15class.shape) # [bs, 15, 512], [bs, 225, 512]
            
            # self.node_features[sample_id] = f_v_15class
            # self.edge_features[sample_id] = f_e_15class
        # del f_v_15class, f_e_15class
        # print('[generate_graph] Waiting for all subprocesses done...')
        print('[generate_graph] Pool, after for loop, time cost:', round(time.time() - start_time, 3), 's')
        p.close()
        print('[generate_graph] Pool, after close, time cost:', round(time.time() - start_time, 3), 's')
        p.join()
        print('[generate_graph] Pool, after join, time cost:', round(time.time() - start_time, 3), 's')
        
        get_ret_start = time.time()
        for ret in results:
            f_v_15class, f_e_15class, sample_id = ret.get()
            self.node_features[sample_id] = f_v_15class
            self.edge_features[sample_id] = f_e_15class
        print('[generate_graph] get_ret time cost:', round(time.time() - get_ret_start, 3), 's')
        print('[generate_graph] All subprocesses done. time cost:', round(time.time() - start_time, 3), 's')
        # print('[generate_graph] self.node_features:', self.node_features, ', self.edge_features:', self.edge_features)
        return self.node_features, self.edge_features
    """
    
    """
    def generate_graph(self, x):
        ''' 不使用多进程，直接通过循环sample_size生成图 融合顺序: 在每一帧上拼接两个人的特征生成图，然后进行时序处理 '''
        sample_size = x.shape[1]
        node_features, edge_features = [], []
        for i in range(sample_size): # 遍历每个sample_size
            one_sample_feature = x[:, i, :, :, :] # 取sample_size个中的一个sample的人脸特征: [bs, c, w, h]
            person_face_1, person_face_2 = one_sample_feature[:, 0:3, :, :], one_sample_feature[:, 3:6, :, :]
            # del one_sample_feature
            # print('[MEFL.py] GraphModel.forward, person_face_1.device: ', person_face_1.device, ', person_face_2.device: ', person_face_2.device)
            
            # concat(person_face_1: [bs, 3, 112, 112], person_face_2: [bs, 3, 112, 112]) -> [bs, 3, 224, 224]
            two_person_face = torch.cat((person_face_1, person_face_2), dim=2)
            # print('[MEFL.py] GraphModel.forward, two_person_face.shape: ', two_person_face.shape) # [bs, 3, 224, 112]
            
            # f_v_15class_1, f_e_15class_1 = self.process_face(person_face_1) # person 1 f_v_15class_1:[bs, 19, 512] f_e_15class_1:[bs, 361, 512]
            # f_v_15class_2, f_e_15class_2 = self.process_face(person_face_2) # person 2 f_v_15class_2:[bs, 19, 512] f_e_15class_2:[bs, 361, 512]
            del person_face_1, person_face_2
            # print('[MEFL.py] GraphModel.forward, f_v_15class_1.shape: ', f_v_15class_1.shape, ', f_e_15class_1.shape: ', f_e_15class_1.shape)
            # print('[MEFL.py] GraphModel.forward, f_v_15class_1.shape: ', f_v_15class_1.shape, ', f_e_15class_1.shape: ', f_e_15class_1.shape, ', f_v_15class_2.shape: ', f_v_15class_2.shape, ', f_e_15class_2.shape: ', f_e_15class_2.shape)
            
            # f_v_15class = torch.cat((f_v_15class_1, f_v_15class_2), dim=1) # [bs, n_nodes(15), 512] -> [bs, n_nodes*2(30), 512] two person
            # f_e_15class = torch.cat((f_e_15class_1, f_e_15class_2), dim=1) # [bs, 361, 512] -> [bs, 722, 512] two person # TODO
            f_v_15class, f_e_15class = self.process_face(two_person_face)
            # print('[MEFL.py] GraphModel.forward, f_v_15class.shape: ', f_v_15class.shape, ', f_e_15class.shape: ', f_e_15class.shape) # [bs, 15, 512], [bs, 225, 512]
            '''
            # cl2, cl_edge2 = self.graph_model(person_face_2)
            # print('[MEFL.py] GraphModel.forward, cl1.shape: ', cl1.shape, ', cl_edge1.shape: ', cl_edge1.shape, ', cl2.shape: ', cl2.shape, ', cl_edge2.shape: ', cl_edge2.shape)
            
            # cl = torch.cat((cl1, cl2), dim=1)
            # cl_edge = torch.cat((cl_edge1, cl_edge2), dim=1)
            # print('[MEFL.py] GraphModel.forward, cl.shape: ', cl.shape, ', cl_edge.shape: ', cl_edge.shape)
            '''
            node_features.append(f_v_15class)
            edge_features.append(f_e_15class)
            # del f_v_15class, f_e_15class, f_v_15class_1, f_e_15class_1, f_v_15class_2, f_e_15class_2
            del f_v_15class, f_e_15class
            if torch.cuda.is_available():
                time_start = time.time()
                gc.collect()
                torch.cuda.empty_cache()
                duration = round(time.time() - time_start, 3)
                if duration > 1:
                    print('[MEFL.py] GraphModel.generate_graph, current sample_size:', i, ', gc.collect() time cost: ', duration, 's')
        return node_features, edge_features
    """
    
    def generate_graph(self, x):
        ''' 不使用多进程，直接通过循环sample_size生成图 融合顺序: 先把单个人的所有帧进行拼接，然后把两个人的图拼接。'''
        sample_size = x.shape[1]
        node_features_1, edge_features_1 = [], []
        node_features_2, edge_features_2 = [], []
        for i in range(sample_size): # 遍历每个sample_size
            one_sample_feature = x[:, i, :, :, :] # 取sample_size个中的一个sample的人脸特征: [bs, c, w, h]
            person_face_1, person_face_2 = one_sample_feature[:, 0:3, :, :], one_sample_feature[:, 3:6, :, :]
            # print('[MEFL.py] GraphModel.forward, person_face_1.device: ', person_face_1.device, ', person_face_2.device: ', person_face_2.device)
            
            f_v_15class_1, f_e_15class_1 = self.process_face(person_face_1) # person 1 f_v_15class_1:[bs, 15, 512] f_e_15class_1:[bs, 225, 512]
            f_v_15class_2, f_e_15class_2 = self.process_face(person_face_2) # person 2 f_v_15class_2:[bs, 15, 512] f_e_15class_2:[bs, 225, 512]
            del person_face_1, person_face_2
            # print('[MEFL.py] GraphModel.forward, f_v_15class_1.shape: ', f_v_15class_1.shape, ', f_e_15class_1.shape: ', f_e_15class_1.shape)
            # print('[MEFL.py] GraphModel.forward, f_v_15class_1.shape: ', f_v_15class_1.shape, ', f_e_15class_1.shape: ', f_e_15class_1.shape, ', f_v_15class_2.shape: ', f_v_15class_2.shape, ', f_e_15class_2.shape: ', f_e_15class_2.shape)
            
            node_features_1.append(f_v_15class_1)
            edge_features_1.append(f_e_15class_1)
            node_features_2.append(f_v_15class_2)
            edge_features_2.append(f_e_15class_2)
            del f_v_15class_1, f_e_15class_1, f_v_15class_2, f_e_15class_2
            
            if torch.cuda.is_available():
                time_start = time.time()
                gc.collect()
                torch.cuda.empty_cache()
                duration = round(time.time() - time_start, 3)
                if duration > 1:
                    print('[MEFL.py] GraphModel.generate_graph, current sample_size:', i, ', gc.collect() time cost: ', duration, 's')
        node_features_1 = torch.stack(node_features_1, dim=1) # [bs, sample_size, 15, 512]
        edge_features_1 = torch.stack(edge_features_1, dim=1) # [bs, sample_size, 225, 512]
        node_features_2 = torch.stack(node_features_2, dim=1) # [bs, sample_size, 15, 512]
        edge_features_2 = torch.stack(edge_features_2, dim=1) # [bs, sample_size, 225, 512]
        # print('[MEFL.py] GraphModel.generate_graph, after stack, node_features_1.shape:', node_features_1.shape, ', edge_features_1.shape:', edge_features_1.shape, ', node_features_2.shape:', node_features_2.shape, ', edge_features_2.shape:', edge_features_2.shape)
        
        node_features = torch.cat((node_features_1, node_features_2), dim=2) # [bs, sample_size, 30, 512]
        edge_features = torch.cat((edge_features_1, edge_features_2), dim=2) # [bs, sample_size, 450, 512]
        # print('[MEFL.py] GraphModel.generate_graph, after cat, node_features.shape:', node_features.shape, ', edge_features.shape:', edge_features.shape)
        return node_features, edge_features

    def forward(self, x):
        forward_start = time.time()
        # print('[MEFL.py] GraphModel.forward, input x.shape:', x.shape) # [bs, sample_size, c, w, h] e.g. [bs, 6, 3, 112, 112]
        node_features, edge_features = self.generate_graph(x)
        sample_size_end = time.time()
        sample_size_loop_duration = round(sample_size_end - forward_start, 3)
        
        # 获得时间序列节点特征和边特征
        # final_node_feature = torch.stack(node_features, dim=1) # [bs, sample_size, 15, 512]
        # final_edge_feature = torch.stack(edge_features, dim=1).mean(dim=1) # [bs, 225, 512] stack smaple_size个[bs, 225, 512] to [bs, sample_size, 225, 512], then [bs, sample_size, 225, 512] -> [bs, 225, 512]
        edge_features = edge_features.mean(dim=1) # [bs, sample_size, num_edges, 512] -> [bs, num_edges, 512]
        # print('final_edge_feature:', final_edge_feature.shape) # [bs, 225, 512]
        # final_edge_feature = self.linear_edge(final_edge_feature.permute(0, 2, 1)).permute(0, 2, 1)
        edge_features = self.linear_edge(edge_features.permute(0, 2, 1)).permute(0, 2, 1) # [bs, 450, 512] -> [bs, 512, 450] -> [bs, 450, 512]
        # print('[MEFL.py] GraphModel.forward, final_node_feature.shape:', final_node_feature.shape, ', final_edge_feature.shape:', final_edge_feature.shape, final_node_feature.device, final_edge_feature.device) # [bs, sample_size, 15, 512], [bs, 900, 512]
        
        adj = edge_features.mean(dim=0) # [bs, num_edges, 512] -> [num_edges, 512]
        # print('[MEFL.py] GraphModel.forward, adj.shape:', adj.shape) # [900, 512]
        num_edges_square, edge_features_dim = adj.shape
        adj = adj.view(int(math.sqrt(num_edges_square)), int(math.sqrt(num_edges_square)), edge_features_dim)
        # print('[MEFL.py] GraphModel.forward, 1-adj.shape:', adj.shape) # [15, 15, 512]
        if not torch.cuda.is_available() and self.cfg.TRAIN.USE_AMP:
            adj = adj.to(torch.float16)
        adj = construct_adj(adj.detach().cpu().numpy(), 3)
        adj = torch.from_numpy(adj).float().to(device)
        # print('[MEFL.py] GraphModel.forward, 2-adj.shape:', adj.shape) # [15*3=45, 15*3=45, 512] because of the 3 in construct_adj(adj.detach().numpy(), 3)
        
        del num_edges_square, edge_features_dim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 时间图序列建模方法1. 使用ctr_gcn方法对时间图序列建模，但是该方法没有考虑边特征  Reference: https://github.com/Uason-Chen/CTR-GCN
        # final_node_feature = final_node_feature.permute(0, 3, 1, 2).unsqueeze(-1) # [bs, 512, sample_size, 15, 1] # 将final_node_feature的维度从 [bs, sample_size, num_nodes, node_features_dim] -> [bs, node_features_dim, sample_size, num_nodes, 1]
        # ret = self.ctr_gcn(final_node_feature)
        # print('[MEFL.py] GraphModel.forward, ret.shape:', ret.shape)
        
        # 时间图序列建模方法2. 使用stsgcn方法对时间图序列建模, 该方法同时考虑了节点特征和边特征 Reference: https://github.com/SmallNana/STSGCN_Pytorch
        output = self.sts_gcn(node_features, adj)
        # print('[MEFL.py] GraphModel.forward, output.shape:', output.shape) # [bs, sample_size, 30]
        if output.shape[1] == 1:
            output = output.squeeze(1) # [bs, 1, 30] -> [bs, 30]   squeeze(1) 是将第1维度为1的维度去掉，即将[bs, 1, 30] -> [bs, 30]
        else:
            output = output.mean(dim=1) # [bs, sample_size, 30] -> [bs, 30]
        # print('[MEFL.py] GraphModel.forward, 2-output:', output.shape, output)
        output = self.linear(output)
        # print('[MEFL.py] GraphModel.forward, after linear, output.shape:', output.shape) # [bs, 30] -> [bs, num_classes]
        
        output = torch.sigmoid(output)
        # print('[MEFL.py] GraphModel.forward, after sigmoid, output:', output.shape, output)
        
        del node_features, adj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('[MEFL.py] GraphModel.forward, iterate sample_size cost:', sample_size_loop_duration, 'seconds, sts_gcn model cost:', round(time.time() - sample_size_end, 3), 'seconds')
        return output

class VisualGraphModel(nn.Module):
    """视觉图模型"""
    def __init__(self, init_weights=True, return_feat=False, num_class=4, pretrained_model=None, backbone='resnet50', sample_size=16, au_class=[12], cfg=None):
        super(VisualGraphModel, self).__init__()
        self.return_feature = return_feat
        assert isinstance(au_class, list) and len(au_class) == 1
        self.au_class = au_class[0]
        self.num_class = num_class
        self.cfg = cfg
        if init_weights:
            initialize_weights(self)
        self.graph_model_12class = MEFARG(num_classes=AU_12CLASS, backbone="resnet50_3d", modal='visual_simple', cfg=self.cfg).to(device)
        self.graph_model_8class = MEFARG(num_classes=AU_8CLASS, backbone="resnet50_3d", modal='visual_simple', cfg=self.cfg).to(device)
        self.fc = nn.Linear(self.au_class, self.num_class)
    
    def forward(self, x):
        print('[VisualGraphModel] input x:', x.shape) # [bs, sample_size, 6(3*2person=6), 112, 112]
        if self.au_class == 12:
            f_v, f_e, cl, cl_edge = self.graph_model_12class(x)
        elif self.au_class == 8:
            f_v, f_e, cl, cl_edge = self.graph_model_8class(x)
        # elif self.au_class == 15: # TODO
        #     f_v, f_e, cl, cl_edge = self.graph_model_12class(x)
        #     f_v, f_e, cl, cl_edge = self.graph_model_8class(x)
        else:
            raise Exception('VisualGraphModel.forward, au_class error!')
        print('[VisualGraphModel] f_v:', f_v.shape, ', f_e:', f_e.shape, ', cl:', cl.shape, ', cl_edge:', cl_edge.shape)
        if self.cfg.MODEL.PREDICTION_FEAT == 'cl':
             x = self.fc(cl)
        elif self.cfg.MODEL.PREDICTION_FEAT == 'cl_edge':
            cl_edge = cl_edge.squeeze(-1)
            self.fc = nn.Linear(cl_edge.shape[-1], self.num_class).to(device)
            x = self.fc(cl_edge)
        else:
            raise Exception('cfg.MODEL.PREDICTION_FEAT error!')
        print('[VisualGraphModel] output x:', x.shape)
        x = torch.sigmoid(x)
        if self.return_feature:
            return x, f_v, f_e, cl, cl_edge
        return x

class AudioGraphModel(nn.Module):
    """ 音频图模型 """
    def __init__(self, init_weights=True, return_feat=False, num_class=4, pretrained_model=None, backbone='resnet50', sample_size=16, au_class=[12], cfg=None):
        super(AudioGraphModel, self).__init__()
        self.return_feature = return_feat
        assert isinstance(au_class, list) and len(au_class) == 1
        self.num_class = num_class
        self.au_class = au_class[0]
        self.cfg = cfg
        if init_weights:
            initialize_weights(self)
        self.audio_graph_model = MEFARG(num_classes=self.au_class, backbone="resnet50", modal='audio').to(device)
        self.fc = nn.Linear(self.au_class, self.num_class)
    
    def forward(self, x):
        print('[AudioGraphModel] input x:', x.shape) # [bs, 2(1*2person=2), 1, 32000(sample_size/5*16000)] 5 is downsample, so duration=samples/5 seconds, sample_rate=16000(num of features per second), so last dim=duration*sample_rate
        f_v, f_e, cl, cl_edge = self.audio_graph_model(x)
        print('[AudioGraphModel] f_v:', f_v.shape, ', f_e:', f_e.shape, ', cl:', cl.shape, ', cl_edge:', cl_edge.shape, 'self.au_class:', self.au_class)
        
        if self.cfg.MODEL.PREDICTION_FEAT == 'cl':
             x = self.fc(cl)
        elif self.cfg.MODEL.PREDICTION_FEAT == 'cl_edge':
            cl_edge = cl_edge.squeeze(-1)
            self.fc = nn.Linear(cl_edge.shape[-1], self.num_class).to(device)
            x = self.fc(cl_edge)
        else:
            raise Exception('cfg.MODEL.PREDICTION_FEAT error!')
        print('[AudioGraphModel] output x:', x.shape)
        x = torch.sigmoid(x)
        if self.return_feature:
            return x, f_v, f_e, cl, cl_edge
        return x

class AudioVisualGraphModel(nn.Module):
    """ 视觉和音频融合的图模型 """
    def __init__(self, init_weights=True, num_class=4, pretrained_model=None, backbone='resnet50', sample_size=16, au_class=[12,12], cfg=None):
        super(AudioVisualGraphModel, self).__init__()
        assert isinstance(au_class, list) and len(au_class) == 2
        self.audio_au_class, self.visual_au_class = au_class[0], au_class[1]
        self.cfg = cfg
        self.num_class = num_class
        if init_weights:
            initialize_weights(self)
        self.audio_graph_model = AudioGraphModel(return_feat=True, num_class=num_class, pretrained_model=pretrained_model, backbone=backbone, sample_size=sample_size, au_class=[self.audio_au_class], cfg=cfg).to(device)
        self.visual_graph_model = VisualGraphModel(return_feat=True, num_class=num_class, pretrained_model=pretrained_model, backbone=backbone, sample_size=sample_size, au_class=[self.visual_au_class], cfg=cfg).to(device)
        self.fc = nn.Linear(self.audio_au_class+self.visual_au_class, num_class)
    
    def forward(self, audio_input, visual_input):
        print('[AudioVisualGraphModel] input audio_input:', audio_input.shape, ', visual_input:', visual_input.shape)
        x_audio, f_v_audio, f_e_audio, cl_audio, cl_edge_audio = self.audio_graph_model(audio_input)
        x_visual, f_v_visual, f_e_visual, cl_visual, cl_edge_visual = self.visual_graph_model(visual_input)
        print('[AudioVisualGraphModel] x_audio:', x_audio.shape, ', f_v_audio:', f_v_audio.shape, ', f_e_audio:', f_e_audio.shape, ', cl_audio:', cl_audio.shape, ', cl_edge_audio:', cl_edge_audio.shape)
        print('[AudioVisualGraphModel] x_visual:', x_visual.shape, ', f_v_visual:', f_v_visual.shape, ', f_e_visual:', f_e_visual.shape, ', cl_visual:', cl_visual.shape, ', cl_edge_visual:', cl_edge_visual.shape)
        
        if self.cfg.MODEL.PREDICTION_FEAT == 'cl':
            x = torch.cat((cl_audio, cl_visual), dim=1)
        elif self.cfg.MODEL.PREDICTION_FEAT == 'cl_edge':
            x = torch.cat((cl_edge_audio.squeeze(-1), cl_edge_visual.squeeze(-1)), dim=1)
        # elif self.cfg.MODEL.PREDICTION_FEAT == 'f_v':
        #     x = torch.cat((f_v_audio, f_v_visual), dim=1) # TODO
        # elif self.cfg.MODEL.PREDICTION_FEAT == 'f_e':
        #     x = torch.cat((f_e_audio, f_e_visual), dim=1) # TODO
        else:
            raise Exception('AudioVisualGraphModel.forward, cfg.MODEL.PREDICTION_FEAT error!')
        print('[AudioVisualGraphModel] after cat, x:', x.shape)
        self.fc = nn.Linear(x.shape[1], self.num_class).to(device)
        x = self.fc(x)
        print('[AudioVisualGraphModel] output x:', x.shape)
        x = torch.sigmoid(x)
        return x

@NETWORK_REGISTRY.register()
def visual_graph_representation_learning(cfg=None): # 视觉图表示学习
    assert cfg.TRAIN.BIMODAL_OPTION == 1, "cfg.TRAIN.BIMODAL_OPTION should be 1 for visual_graph_representation_learning"
    if cfg.TRAIN.PRE_TRAINED_MODEL:
        # 如果 cfg.TRAIN.PRE_TRAINED_MODEL中包含分号; 则构建一个列表，以分号分隔，将多个预训练模型的路径保存到列表中
        if ';' in cfg.TRAIN.PRE_TRAINED_MODEL:
            pretrained_model_path = cfg.TRAIN.PRE_TRAINED_MODEL.split(';')
        else:
            pretrained_model_path = cfg.TRAIN.PRE_TRAINED_MODEL
    else:
        pretrained_model_path = None
    # print('[visual_graph_representation_learning] pretrained_model_path: ', pretrained_model_path)
    
    if 'resnet50' in cfg.TRAIN.PRE_TRAINED_MODEL:
        backbone_net = "resnet50"
    elif 'swin_tiny' in cfg.TRAIN.PRE_TRAINED_MODEL:
        backbone_net = "swin_transformer_tiny"
    
    # if 'BP4D' in cfg.TRAIN.PRE_TRAINED_MODEL:
    #     num_class = 12
    # elif 'DISFA' in cfg.TRAIN.PRE_TRAINED_MODEL:
    #     num_class = 8
    # print('[visual_graph_representation_learning] num_class: ', num_class, ', backbone_net: ', backbone_net)
    
    if cfg.MODEL.BACKBONE_INPUT == 'frame': # 方法1. 输入MEFARG backbone的数据是每一帧图像
        graph_model = GraphModel(num_class=cfg.MODEL.NUM_CLASS, pretrained_model=pretrained_model_path, backbone=backbone_net, sample_size=cfg.DATA.SAMPLE_SIZE, au_class=cfg.MODEL.AU_CLASS, cfg=cfg)
    elif cfg.MODEL.BACKBONE_INPUT == 'video': # 方法2. 输入MEFARG backbone的数据是每一段视频，即图像序列
        graph_model = VisualGraphModel(num_class=cfg.MODEL.NUM_CLASS, pretrained_model=pretrained_model_path, backbone=backbone_net, sample_size=cfg.DATA.SAMPLE_SIZE, au_class=cfg.MODEL.AU_CLASS, cfg=cfg)
    
    if torch.cuda.is_available() and cfg.TRAIN.USE_HALF:
        graph_model.half()
    graph_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return graph_model

@NETWORK_REGISTRY.register()
def audio_graph_representation_learning(cfg=None): # 音频图表示学习
    assert cfg.TRAIN.BIMODAL_OPTION == 2, "cfg.TRAIN.BIMODAL_OPTION should be 2 for audio_graph_representation_learning"
    if cfg.TRAIN.PRE_TRAINED_MODEL:
        # 如果 cfg.TRAIN.PRE_TRAINED_MODEL中包含分号; 则构建一个列表，以分号分隔，将多个预训练模型的路径保存到列表中
        if ';' in cfg.TRAIN.PRE_TRAINED_MODEL:
            pretrained_model_path = cfg.TRAIN.PRE_TRAINED_MODEL.split(';')
        else:
            pretrained_model_path = cfg.TRAIN.PRE_TRAINED_MODEL
    else:
        pretrained_model_path = None
    # print('[visual_graph_representation_learning] pretrained_model_path: ', pretrained_model_path)
    
    if 'resnet50' in cfg.TRAIN.PRE_TRAINED_MODEL:
        backbone_net = "resnet50"
    elif 'swin_tiny' in cfg.TRAIN.PRE_TRAINED_MODEL:
        backbone_net = "swin_transformer_tiny"
    
    graph_model = AudioGraphModel(num_class=cfg.MODEL.NUM_CLASS, pretrained_model=pretrained_model_path, backbone=backbone_net, sample_size=cfg.DATA.SAMPLE_SIZE, au_class=cfg.MODEL.AU_CLASS, cfg=cfg)
    if torch.cuda.is_available() and cfg.TRAIN.USE_HALF:
        graph_model.half()
    graph_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return graph_model

@NETWORK_REGISTRY.register()
def audiovisual_graph_representation_learning(cfg=None): # 视觉+音频融合 图表示学习
    assert cfg.TRAIN.BIMODAL_OPTION == 3, "cfg.TRAIN.BIMODAL_OPTION should be 3 for audiovisual_graph_representation_learning"
    if cfg.TRAIN.PRE_TRAINED_MODEL:
        # 如果 cfg.TRAIN.PRE_TRAINED_MODEL中包含分号; 则构建一个列表，以分号分隔，将多个预训练模型的路径保存到列表中
        if ';' in cfg.TRAIN.PRE_TRAINED_MODEL:
            pretrained_model_path = cfg.TRAIN.PRE_TRAINED_MODEL.split(';')
        else:
            pretrained_model_path = cfg.TRAIN.PRE_TRAINED_MODEL
    else:
        pretrained_model_path = None
    # print('[visual_graph_representation_learning] pretrained_model_path: ', pretrained_model_path)
    
    if 'resnet50' in cfg.TRAIN.PRE_TRAINED_MODEL:
        backbone_net = "resnet50"
    elif 'swin_tiny' in cfg.TRAIN.PRE_TRAINED_MODEL:
        backbone_net = "swin_transformer_tiny"
    
    graph_model = AudioVisualGraphModel(num_class=cfg.MODEL.NUM_CLASS, pretrained_model=pretrained_model_path, backbone=backbone_net, sample_size=cfg.DATA.SAMPLE_SIZE, au_class=cfg.MODEL.AU_CLASS, cfg=cfg)
    if torch.cuda.is_available() and cfg.TRAIN.USE_HALF:
        graph_model.half()
    graph_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return graph_model