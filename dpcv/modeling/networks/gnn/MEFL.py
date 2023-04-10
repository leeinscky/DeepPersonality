import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
from .resnet import resnet18, resnet50, resnet101
from .graph import create_e_matrix
from .graph_edge_model import GEM
from .basic_block import *
from dpcv.modeling.module.weight_init_helper import initialize_weights
from dpcv.modeling.networks.build import NETWORK_REGISTRY
from .MEFL_gratis import MEFARG as MEFARG_GRATIS

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
        print('[MEFL.py] GNN, input x.shape:', x.shape, ', edge.shape:', edge.shape) # x.shape: [bs, 12, 512], edge.shape: [bs, 144, 512] 因为BP4D有12个节点，144个边
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
        print('[MEFL.py] GNN, output x.shape:', x.shape, ', edge.shape:', edge.shape)
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
        self.edge_fc = nn.Linear(self.in_channels, 4)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.edge_fc.weight)
        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        print('\n[MEFL.py] class Head.forward, input x.shape: ', x.shape) # x.shape: [bs, 49, 512]
        
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)
        print('[MEFL.py] class Head.forward, f_u.shape: ', f_u.shape, 'f_v.shape: ', f_v.shape) # f_u.shape: [bs, 12, 49, 512], f_v.shape: [bs, 12, 512]

        # MEFL
        f_e = self.edge_extractor(f_u, x)
        print('[MEFL.py] class Head.forward, edge_extractor 的输入参数 f_u.shape: ', f_u.shape, 'x.shape: ', x.shape, ', 计算结果 f_e.shape: ', f_e.shape) 
        # f_u.shape: [bs, 12, 49, 512], x.shape: [bs, 49, 512], 计算结果 f_e.shape: [bs, 144, 49, 512]
        f_e = f_e.mean(dim=-2)
        
        print('[MEFL.py] class Head.forward, GNN的输入参数, f_v.shape: ', f_v.shape, 'f_e.shape: ', f_e.shape) # f_v.shape: [bs, 12, 512], f_e.shape: [bs, 144, 512]
        f_v, f_e = self.gnn(f_v, f_e)
        print('[MEFL.py] class Head.forward, after self.gnn, f_v.shape: ', f_v.shape, 'f_e.shape: ', f_e.shape) # f_v.shape: [bs, 12, 512], f_e.shape: [bs, 144, 512]

        b, n, c = f_v.shape
        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1) # cl means class logits
        print('[MEFL.py] class Head.forward, after F.normalize, cl.shape: ', cl.shape, 'sc.shape: ', sc.shape) # cl.shape: [bs, 12, 512], sc.shape: [12, 512]
        
        cl = (cl * sc.view(1, n, c)).sum(dim=-1, keepdim=False)
        cl_edge = self.edge_fc(f_e)
        print('[MEFL.py] class Head.forward, final return cl.shape: ', cl.shape, 'cl_edge.shape: ', cl_edge.shape) # cl.shape: [bs, 12], cl_edge.shape: [bs, 144, 4]
        return cl, cl_edge, f_v, f_e


class MEFARG(nn.Module):
    def __init__(self, num_classes=12, backbone='swin_transformer_base'):
        super(MEFARG, self).__init__()
        print('[MEFL.py] class MEFARG.__init__ backbone: ', backbone)
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
                # self.backbone = resnet50() # pretrained=True
                self.backbone = resnet50(pretrained=False) # pretrained=False
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, num_classes)

    def forward(self, x):
        # x: b d c
        # x = torch.randn(2, 3, 112, 112) # TODO temp test
        # x = torch.randn(2, 3, 224, 224) # TODO temp test
        print('[MEFL.py] MEFARG.forward, input x.shape: ', x.shape) # 3 channels: x.shape=[bs, 3, 112, 112]; 6 channels: x.shape=[bs, 6, 112, 112]
        x = self.backbone(x)
        print('[MEFL.py] MEFARG.forward, after backbone, x.shape: ', x.shape) # 3 channels: x.shape=[bs, 16, 2048]; 6 channels: x.shape=[bs, 16, 2048]
        x = self.global_linear(x)
        print('[MEFL.py] MEFARG.forward, after global_linear, x.shape: ', x.shape) # 3 channels: x.shape=[bs, 16, 512]; 6 channels: x.shape=[bs, 16, 512]
        cl, cl_edge, f_v, f_e = self.head(x)
        print('[MEFL.py] MEFARG.forward, after head, cl.shape: ', cl.shape, 'cl_edge.shape: ', cl_edge.shape) # cl.shape: [bs, num_class], cl_edge.shape: [bs, num_class*num_class, num_class]
        return cl, cl_edge, f_v, f_e


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


def union_au_class(au_12class, au_8class):
    """     
    au_12class: [bs, 12, 512]
    au_8class: [bs, 8, 512]
    return: [bs, 15, 512] 
    """
    
    bs = au_12class.shape[0]
    au_15class = torch.zeros(bs, 15, au_12class.shape[2])

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
    
    # simplified
    au_15class[:, :5, :] = au_12class[:, :5, :] # AU1-AU7         au_15class的第1到第5列
    au_15class[:, 5, :] = au_8class[:, 5, :] # AU9                au_15class的第6列
    au_15class[:, 6:13, :] = au_12class[:, 5:12, :] # AU10-AU24   au_15class的第7到第13列 au_15class[:, 6:13] 是指 au_15class 在第二维中从索引 6 开始（包括第 6 列），到索引 13 结束（不包括第 13 列）的子数组。换句话说，它包含 au_15class 中的第 7-13 列。
    au_15class[:, 13:15, :] = au_8class[:, 6:8, :] # AU25-AU26    au_15class的第14到第15列
    
    # print('au_12class: ', au_12class, '\nau_8class: ', au_8class, '\nau_15class: ', au_15class)
    return au_15class


def process_node(au_15class, au_4class):
    print('[MEFL.py] process_node, input au_15class.shape: ', au_15class.shape, 'au_4class.shape: ', au_4class.shape) # au_15class.shape: [bs, 15, 512] au_4class.shape: [bs, 4, 512]
    # [bs, 15, 512], [bs, 4, 512] -> [bs, 19, 512]
    au_19class = torch.cat((au_15class, au_4class), dim=1)
    return au_19class

class GraphModel(nn.Module):
    def __init__(self, init_weights=True, num_class=2, pretrained_model=None, backbone='resnet50'):
        super(GraphModel, self).__init__()

        if init_weights:
            initialize_weights(self)
        
        ### one graph model
        # self.graph_model = MEFARG(num_classes=num_class, backbone=backbone)
        # if pretrained_model is not None:
        #     self.graph_model = load_state_dict(self.graph_model, pretrained_model)
        
        ### two graph model, 12 class + 8 class, then obtain the union: a totoal of 15 AU class 取并集获得15个AU, refer to the paper: https://arxiv.org/pdf/2205.01782.pdf
        self.graph_model_12class = MEFARG(num_classes=12, backbone="swin_transformer_base") # backbone="swin_transformer_tiny" 和 backbone="swin_transformer_small" 时，输出的特征维度都是384；当backbone="swin_transformer_base"时，输出的特征维度是512
        self.graph_model_8class = MEFARG(num_classes=8, backbone="resnet50")
        ### graph model for 4 nodes features, refer to the paper: https://arxiv.org/pdf/2211.12482.pdf
        self.graph_model_4class = MEFARG_GRATIS(num_classes=4, backbone="resnet50")
        
        if pretrained_model is not None:
            print('[MEFL.py] MEFARG load pretrained model: ', pretrained_model)
            # self.graph_model_12class = load_state_dict(self.graph_model_12class, pretrained_model[0])
            self.graph_model_8class = load_state_dict(self.graph_model_8class, pretrained_model[1])

    def forward(self, x):
        print('[MEFL.py] GraphModel.forward, input x.shape: ', x.shape) # [bs, sample_size, c, w, h] e.g. [bs, 6, 3, 112, 112]
        # 遍历每个sample_size
        for i in range(x.shape[1]):
            one_sample_feature =  x[:, i, :, :, :] # 取sample_size个中的一个sample的人脸特征: [bs, c, w, h]
            person_face_1, person_face_2 = one_sample_feature[:, 0:3, :, :], one_sample_feature[:, 3:6, :, :]
            print('[MEFL.py] GraphModel.forward, one_sample_feature.shape: ', one_sample_feature.shape, ', person_face_1.shape:', person_face_1.shape) # one_sample_feature.shape:  torch.Size([bs, channels, w, h]) , person_face_1.shape: torch.Size([bs, channels/2, w, h])
            
            # cl, cl_edge = self.graph_model(one_sample_feature)
            cl1_12class, cl_edge1_12class, f_v_12class, f_e_12class = self.graph_model_12class(person_face_1)
            cl1_8class, cl_edge1_8class, f_v_8class, f_e_8class = self.graph_model_8class(person_face_1)
            # print('[MEFL.py] GraphModel.forward, cl1_12class.shape: ', cl1_12class.shape, ', cl1_8class.shape: ', cl1_8class.shape, ', cl_edge1_12class.shape: ', cl_edge1_12class.shape, ', cl_edge1_8class.shape: ', cl_edge1_8class.shape) # cl1_12class.shape:[bs, 12], cl1_8class.shape:[bs, 8], cl_edge1_12class.shape:[bs, 144, 4], cl_edge1_8class.shape:[bs, 64, 4]
            print('[MEFL.py] GraphModel.forward, f_v_12class.shape: ', f_v_12class.shape, ', f_e_12class.shape: ', f_e_12class.shape, ', f_v_8class.shape: ', f_v_8class.shape, ', f_e_8class.shape: ', f_e_8class.shape) 
            f_v_15class = union_au_class(f_v_12class, f_v_8class)
            print('[MEFL.py] GraphModel.forward, f_v_15class.shape: ', f_v_15class.shape)
            
            cl1_4class, cl_edge1_4class, f_v_4class, f_e_4class = self.graph_model_4class(person_face_1)
            print('[MEFL.py] GraphModel.forward, f_v_4class.shape: ', f_v_4class.shape)
            
            ### process node features 15 class + 4 class, and get the final node features 19 class
            f_v_19class = process_node(f_v_15class, f_v_4class)
            print('[MEFL.py] GraphModel.forward, f_v_19class.shape: ', f_v_19class.shape)
            
            # cl2, cl_edge2 = self.graph_model(person_face_2)
            # print('[MEFL.py] GraphModel.forward, cl1.shape: ', cl1.shape, ', cl_edge1.shape: ', cl_edge1.shape, ', cl2.shape: ', cl2.shape, ', cl_edge2.shape: ', cl_edge2.shape)
            
            # cl = torch.cat((cl1, cl2), dim=1)
            # cl_edge = torch.cat((cl_edge1, cl_edge2), dim=1)
            # print('[MEFL.py] GraphModel.forward, cl.shape: ', cl.shape, ', cl_edge.shape: ', cl_edge.shape)
        return f_v_19class


@NETWORK_REGISTRY.register()
def visual_graph_representation_learning(cfg=None): # 视觉图表示学习
    if cfg.TRAIN.PRE_TRAINED_MODEL:
        # 如果 cfg.TRAIN.PRE_TRAINED_MODEL中包含分号; 则构建一个列表，以分号分隔，将多个预训练模型的路径保存到列表中
        if ';' in cfg.TRAIN.PRE_TRAINED_MODEL:
            pretrained_model_path = cfg.TRAIN.PRE_TRAINED_MODEL.split(';')
        else:
            pretrained_model_path = cfg.TRAIN.PRE_TRAINED_MODEL
    else:
        pretrained_model_path = None
    print('[visual_graph_representation_learning] pretrained_model_path: ', pretrained_model_path)
    
    if 'resnet50' in cfg.TRAIN.PRE_TRAINED_MODEL:
        backbone_net = "resnet50"
    elif 'swin_tiny' in cfg.TRAIN.PRE_TRAINED_MODEL:
        backbone_net = "swin_transformer_tiny"
    
    if 'BP4D' in cfg.TRAIN.PRE_TRAINED_MODEL:
        num_class = 12
    elif 'DISFA' in cfg.TRAIN.PRE_TRAINED_MODEL:
        num_class = 8
    # num_class = cfg.MODEL.NUM_CLASS
    # print('[visual_graph_representation_learning] num_class: ', num_class, ', backbone_net: ', backbone_net)
    
    graph_model = GraphModel(num_class=num_class, pretrained_model=pretrained_model_path, backbone=backbone_net)
    graph_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return graph_model