import torch
import torch.nn.functional as F
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_half = False

class gcn_operation(nn.Module):
    def __init__(self, adj, in_dim, out_dim, num_vertices, activation='GLU'):
        """
        图卷积模块
        :param adj: 邻接图
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param num_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(gcn_operation, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation

        assert self.activation in {'GLU', 'relu'}

        if self.activation == 'GLU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True) # 64 -> 128
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True) # 64 -> 64
        # print('[gcn_operation] in_dim:', in_dim, 'out_dim:', out_dim, 'num_vertices:', num_vertices, 'activation:', activation) # in_dim: 64 out_dim: 64 num_vertices: 307 activation: GLU
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask=None, adj=None):
        """
        :param x: (3*N, B, Cin)
        :param mask:(3*N, 3*N)
        :return: (3*N, B, Cout)
        """
        # print('[stsgcn.py]gcn_operation x:', x.shape) # [45, 2, 64]
        if adj is not None:
            self.adj = adj
        
        adj = self.adj
        # print('[stsgcn.py]gcn_operation adj:', adj.shape) # [45, 45, 512]
        if mask is not None:
            adj = adj.to(mask.device) * mask
        
        # [3*N, 3*N, Cin] -> [3*N, 3*N, 1] -> [3*N, 3*N]
        adj = adj.mean(dim=-1).squeeze(dim=-1)
        # print('[stsgcn.py]gcn_operation adj:', adj, '\nx:', x) # adj中元素：1, 0, 0-1之间的小数以及部分很大的数  x中元素：0, 0-1之间的小数, 以及部分很大的数，这导致 后面的计算也会出现大数
        # print('[stsgcn.py]gcn_operation adj:', adj.shape, 'x: ', x.shape) # adj:[57, 57, 512] x:[57, 1, 64]
        # print('[stsgcn.py]gcn_operation adj.dtype:', adj.dtype, 'x.dtype:', x.dtype)
        if torch.cuda.is_available() and use_half:
            adj = adj.half()
            x = x.half()
        x = torch.einsum('nm, mbc->nbc', adj.to(x.device), x)  # 3*N, B, Cin
        # x = torch.einsum('nmk, mbc->nbc', adj.to(x.device), x)
        # print('[stsgcn.py]gcn_operation after einsum, x:', x.shape) #  x: [57, 1, 64] 很多1000-2000之间的数   # TODO 这一步是出现极大值的根源？？

        if self.activation == 'GLU':
            lhs_rhs = self.FC(x)  # 3*N, B, 2*Cout
            # print('[stsgcn.py]gcn_operation GLU lhs_rhs:', lhs_rhs.shape) # lhs_rhs: [921, 32, 128]  lhs_rhs中很多 -4024, 1778.7834 之类的数
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # 3*N, B, Cout
            # print('[stsgcn.py]gcn_operation GLU lhs:', lhs.shape) # 1694054 -7036545 之类的数
            # print('[stsgcn.py]gcn_operation GLU rhs:', rhs.shape) # 12955459 7311653 14938745 之类的数
            # print('[stsgcn.py]gcn_operation GLU torch.sigmoid(rhs):', torch.sigmoid(rhs)) # 全是1

            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs
            
            # print('[stsgcn.py]gcn_operation GLU out:', out.shape) # out: [57（3*N）, 1, 64] 18299654144 和 0 之类的数
            return out

        elif self.activation == 'relu':
            # print('[stsgcn.py]gcn_operation relu self.FC(x):', self.FC(x).shape) # x: [921, 32, 64]
            return self.relu(self.FC(x))  # 3*N, B, Cout


class STSGCM(nn.Module):
    def __init__(self, adj, in_dim, out_dims, num_of_vertices, activation='GLU'):
        """
        :param adj: 邻接矩阵
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(STSGCM, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation

        self.gcn_operations = nn.ModuleList()

        self.gcn_operations.append(
            gcn_operation(
                adj=self.adj,
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                num_vertices=self.num_of_vertices,
                activation=self.activation
            )
        )

        for i in range(1, len(self.out_dims)):
            self.gcn_operations.append(
                gcn_operation(
                    adj=self.adj,
                    in_dim=self.out_dims[i-1],
                    out_dim=self.out_dims[i],
                    num_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )

    def forward(self, x, mask=None, adj=None):
        """
        :param x: (3N, B, Cin)
        :param mask: (3N, 3N)
        :return: (N, B, Cout)
        """
        need_concat = []
        # print('[stsgcn.py]STSGCM input x:', x.shape, x)
        
        for i in range(len(self.out_dims)):
            x = self.gcn_operations[i](x, mask, adj)
            # print('[stsgcn.py]STSGCM after gcn_operations, x:', x.shape) # x 中出现了-5060752， 0 之类的数值
            need_concat.append(x)

        # shape of each element is (1, N, B, Cout)
        need_concat = [
            torch.unsqueeze(
                h[self.num_of_vertices: 2 * self.num_of_vertices], dim=0
            ) for h in need_concat
        ]

        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values  # (N, B, Cout)
        # print('[stsgcn.py]STSGCM out:', out.shape)

        del need_concat

        return out


class STSGCL(nn.Module):
    def __init__(self,
                 adj,
                 history,
                 num_of_vertices,
                 in_dim,
                 out_dims,
                 strides=3,
                 activation='GLU',
                 temporal_emb=True,
                 spatial_emb=True):
        """
        :param adj: 邻接矩阵
        :param history: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        :param temporal_emb: 加入时间位置嵌入向量
        :param spatial_emb: 加入空间位置嵌入向量
        """
        super(STSGCL, self).__init__()
        self.adj = adj
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices

        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        self.STSGCMS = nn.ModuleList()
        for i in range(self.history - self.strides + 1):
            self.STSGCMS.append(
                STSGCM(
                    adj=self.adj,
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )

        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.history, 1, self.in_dim))
            # 1, T, 1, Cin

        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))
            # 1, 1, N, Cin

        self.bn = nn.BatchNorm2d(history-2)
        self.reset()

    def reset(self):
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)

        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, mask=None, adj=None):
        """
        :param x: B, T, N, Cin
        :param mask: (N, N)
        :return: B, T-2, N, Cout
        """
        # print('[stsgcn.py]STSGCL input x:', x.shape)
        if self.temporal_emb:
            x = x + self.temporal_embedding

        if self.spatial_emb:
            x = x + self.spatial_embedding

        need_concat = []
        batch_size = x.shape[0]
        
        # print('[stsgcn.py]STSGCL x:', x.shape, x) # [bs, 12, 19, 64] x中主要有0,0-1之间的小数, 1626229，以及部分大数 这类数值
        
        for i in range(self.history - self.strides + 1):
            t = x[:, i: i+self.strides, :, :]  # (B, 3, N, Cin)
            # print('[stsgcn.py]STSGCL 1-t:', t.shape) # [bs, 3, 19, 64], t中主要有0, 0-1之间的小数这2类数值

            t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim]) # (B, 3*N, Cin)
            # print('[stsgcn.py]STSGCL 2-t:', t.shape) # [bs, 57, 64], t中主要有0, 0-1.1之间的小数这2类数值

            t = self.STSGCMS[i](t.permute(1, 0, 2), mask, adj) # (3*N, B, Cin) -> (N, B, Cout)
            # print('[stsgcn.py]STSGCL 3-t:', t.shape) # [19, bs, 64], t中主要有 5972860928,1892,0 这类数值  TODO

            t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)  # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)
            # print('[stsgcn.py]STSGCL 4-t:', t.shape) # t中元素：3419332753688579390465638400 或者 -0
            need_concat.append(t)

        out = torch.cat(need_concat, dim=1)  # (B, T-2, N, Cout) e.g. (bs, 10, 19, 64), (bs, 8, 19, 64), (bs, 6, 19, 64), (bs, 4, 19, 64)
        
        self.bn = nn.BatchNorm2d(out.shape[1]).to(out.device)
        out = self.bn(out)

        del need_concat, batch_size
        # print('[stsgcn.py]STSGCL out:', out.shape, out) # out中有5972860928 这种元素值，导致后面出现nan
        return out


class output_layer(nn.Module):
    def __init__(self, num_of_vertices, history, in_dim,
                 hidden_dim=128, horizon=12):
        """
        预测层，注意在作者的实验中是对每一个预测时间step做处理的，也即他会令horizon=1
        :param num_of_vertices:节点数
        :param history:输入时间步长
        :param in_dim: 输入维度
        :param hidden_dim:中间层维度
        :param horizon:预测时间步长
        """
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.history = history
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        # print('[stsgcn.py]output_layer init__ in_dim:', self.in_dim, 'history:', self.history, 'hidden_dim:', self.hidden_dim, 'horizon:', self.horizon) # in_dim: 64 history: 4 hidden_dim: 128 horizon: 1
        self.FC1 = nn.Linear(self.in_dim * self.history, self.hidden_dim, bias=True) # 64*4, 128

        self.FC2 = nn.Linear(self.hidden_dim, self.horizon, bias=True) # 128, 1
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: (B, Tin, N, Cin)
        :return: (B, Tout, N)
        """
        batch_size = x.shape[0]

        x = x.permute(0, 2, 1, 3)  # B, N, Tin, Cin  [bs, 19, 4, 64]
        # print('[stsgcn.py]output_layer x.shape:', x.shape, self.FC1(x.reshape(batch_size, self.num_of_vertices, -1))) # self.FC1(x.reshape(batch_size, self.num_of_vertices, -1)) 里的元素出现了nan

        x = self.FC1(x.reshape(batch_size, self.num_of_vertices, -1))
        x = self.relu(x)
        # (B, N, Tin, Cin) -> (B, N, Tin * Cin) -> (B, N, hidden)
        # print('[stsgcn.py]output_layer out1.shape:', out1.shape, out1) # [bs, 19, 128] out1中的元素出现了nan

        x = self.FC2(x)  # (B, N, hidden) -> (B, N, horizon)
        # print('[stsgcn.py]output_layer out2.shape:', out2.shape, out2) # [bs, 19, 1] out2中的元素出现了nan

        del batch_size

        return x.permute(0, 2, 1)  # B, horizon, N


class STSGCN(nn.Module):
    def __init__(self, adj, history, num_of_vertices, in_dim, hidden_dims,
                 first_layer_embedding_size, out_layer_dim, activation='GLU', use_mask=True,
                 temporal_emb=True, spatial_emb=True, horizon=12, strides=3):
        """

        :param adj: local时空间矩阵
        :param history:输入时间步长
        :param num_of_vertices:节点数量
        :param in_dim:输入维度
        :param hidden_dims: lists, 中间各STSGCL层的卷积操作维度
        :param first_layer_embedding_size: 第一层输入层的维度
        :param out_layer_dim: 输出模块中间层维度
        :param activation: 激活函数 {relu, GlU}
        :param use_mask: 是否使用mask矩阵对adj进行优化
        :param temporal_emb:是否使用时间嵌入向量
        :param spatial_emb:是否使用空间嵌入向量
        :param horizon:预测时间步长
        :param strides:滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        """
        super(STSGCN, self).__init__()
        # 分行打印所有参数
        print("STSGCN params: adj.shape: ", adj.shape, " history: ", history, " num_of_vertices: ", num_of_vertices, ' in_dim: ', in_dim, \
            ' hidden_dims: ', hidden_dims, ' first_layer_embedding_size: ', first_layer_embedding_size, ' out_layer_dim: ', out_layer_dim, ' activation: ', activation, \
            ' use_mask: ', use_mask, ' temporal_emb: ', temporal_emb, ' spatial_emb: ', spatial_emb, ' horizon: ', horizon, ' strides: ', strides)
        # STSGCN params: adj.shape:  torch.Size([921, 921])  history: 12  num_of_vertices:  307  in_dim:  1  hidden_dims:  [[64, 64, 64], [64, 64, 64], [64, 64, 64], [64, 64, 64]]  first_layer_embedding_size:  64  out_layer_dim:  128  activation:  GLU  use_mask:  True  temporal_emb:  True  spatial_emb:  True  horizon:  12  strides:  3

        self.adj = adj
        self.num_of_vertices = num_of_vertices
        self.hidden_dims = hidden_dims
        self.out_layer_dim = out_layer_dim
        self.activation = activation
        self.use_mask = use_mask

        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.horizon = horizon
        self.strides = strides

        self.First_FC = nn.Linear(in_dim, first_layer_embedding_size, bias=True)
        self.STSGCLS = nn.ModuleList()
        self.STSGCLS.append(
            STSGCL(
                adj=self.adj,
                history=history,
                num_of_vertices=self.num_of_vertices,
                in_dim=first_layer_embedding_size,
                out_dims=self.hidden_dims[0],
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb
            )
        )

        # print('[STSGCN] 1-in_dim:', in_dim, 'history:', history)
        in_dim = self.hidden_dims[0][-1]
        history -= (self.strides - 1)
        # print('[STSGCN] 2-in_dim:', in_dim, 'history:', history)

        for idx, hidden_list in enumerate(self.hidden_dims):
            if idx == 0:
                continue
            self.STSGCLS.append(
                STSGCL(
                    adj=self.adj,
                    history=history,
                    num_of_vertices=self.num_of_vertices,
                    in_dim=in_dim,
                    out_dims=hidden_list,
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb
                )
            )
            history -= (self.strides - 1)
            in_dim = hidden_list[-1]
            # print('[STSGCN] 3-in_dim:', in_dim, 'history:', history)
        
        self.predictLayer = nn.ModuleList()
        for t in range(self.horizon):
            self.predictLayer.append(
                output_layer(
                    num_of_vertices=self.num_of_vertices,
                    history=history,
                    in_dim=in_dim,
                    hidden_dim=out_layer_dim,
                    horizon=1
                )
            )

        if self.use_mask:
            mask = torch.zeros_like(self.adj)
            mask[self.adj != 0] = self.adj[self.adj != 0]
            self.mask = nn.Parameter(mask)
        else:
            self.mask = None
        self.relu = nn.ReLU(inplace=True)
        # print('[stsgcn.py]STSGCN init self.STSGCLS:', self.STSGCLS)

    def forward(self, x, adj):
        """
        :param x: B, Tin, N, Cin) # batch_size, time_step, num_of_vertices, in_dim
        :return: B, Tout, N # batch_size, time_step, num_of_vertices
        """
        # print('[stsgcn.py]STSGCN input x:', x.shape, x.dtype) # [bs, sample_size, 19, 512] x中的元素基本都在0-3之间
        if torch.cuda.is_available() and use_half:
            x = x.half()
        # print('[stsgcn.py]STSGCN self.First_FC:', next(self.First_FC.parameters()).dtype)
        x = self.relu(self.First_FC(x))  # B, Tin, N, Cin
        # print('[stsgcn.py]STSGCN after fist FC, x:', x.shape) # [bs, sample_size, 19, 64] x中的元素基本都在0-1之间
        
        for model in self.STSGCLS:
            x = model(x, self.mask, adj) # (B, T - 8, N, Cout)
            # print('[stsgcn.py]STSGCN after STSGCL, x.shape:', x.shape) # [bs, 10, 19, 64], [bs, 8, 19, 64], [bs, 6, 19, 64], [bs, 4, 19, 64] x中的元素基本都在0-5972860928左右，即出现很大的数值
            # 注意：出现nan的根源在这里，因为x中的元素基本都在0-5972860928左右，即出现很大的数值，导致后面 predictLayer 的计算出现nan
        
        need_concat = []
        for i in range(self.horizon):
            out_step = self.predictLayer[i](x)  # (B, 1, N)
            # print('[stsgcn.py]STSGCN after output layer, out_step:', out_step.shape) # [bs, 1, 19]   out_step中的元素全是nan
            need_concat.append(out_step)

        out = torch.cat(need_concat, dim=1)  # B, Tout, N
        # print('[stsgcn.py]STSGCN after concat, out:', out.shape) # [bs, 12, 19]

        del need_concat

        return out

