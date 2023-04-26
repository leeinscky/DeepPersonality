import sys
import numpy as np

sys.path.extend(['../'])
# from graph import tools

# num_node = 20 # original 20 nodes
num_node = 19
self_link = [(i, i) for i in range(num_node)]
# inward_ori_index = [(1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6),
#                     (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (13, 1),
#                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
#                     (20, 19)] # original 20 nodes # Refer: https://github.com/Uason-Chen/CTR-GCN/issues/12
inward_ori_index = [(1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6),
                    (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward
print('len(self_link):', len(self_link), 'len(inward):', len(inward), 'len(outward):', len(outward), 'len(neighbor):', len(neighbor))
# len(self_link): 19 len(inward): 19 len(outward): 19 len(neighbor): 38

# def get_sgp_mat(num_in, num_out, link):
#     # 生成一个有向图的邻接矩阵，矩阵中元素A[i][j]表示从节点i到节点j是否有边，1表示有边，0表示没有边。
#     A = np.zeros((num_in, num_out))
#     for i, j in link:
#         A[i, j] = 1
#     A_norm = A / np.sum(A, axis=0, keepdims=True)
#     return A_norm


# def get_k_scale_graph(scale, A):
#     # 将邻接矩阵进行k次幂运算后，将非零元素赋值为1，得到新的邻接矩阵。
#     if scale == 1:
#         return A
#     An = np.zeros_like(A)
#     A_power = np.eye(A.shape[0])
#     for k in range(scale):
#         A_power = A_power @ A
#         An += A_power
#     An[An > 0] = 1
#     return An


# def normalize_adjacency_matrix(A):
#     # 对于无向图，将邻接矩阵进行归一化，得到新的邻接矩阵。对于邻接矩阵A，新的邻接矩阵A_norm[i][j] = A[i][j] / sqrt(sum(A[i][k]) * sum(A[j][k]))，其中k是所有与i和j相邻的节点。
#     node_degrees = A.sum(-1)
#     degs_inv_sqrt = np.power(node_degrees, -0.5)
#     norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
#     return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


# def k_adjacency(A, k, with_self=False, self_factor=1):
#     # 对邻接矩阵进行k次幂运算，得到新的邻接矩阵。对于邻接矩阵A，新的邻接矩阵A^k[i][j]表示i到j的k步最短路路径数。如果with_self=True，则A^k[i][i]会加上self_factor，self_factor默认为1。
#     assert isinstance(A, np.ndarray)
#     I = np.eye(len(A), dtype=A.dtype)
#     if k == 0:
#         return I
#     Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
#        - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
#     if with_self:
#         Ak += (self_factor * I)
#     return Ak


# def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
#     # 生成一个包含自环、入边、出边、入边的二阶邻居和出边的二阶邻居的五维邻接矩阵。
#     I = edge2mat(self_link, num_node)
#     A1 = edge2mat(inward, num_node)
#     A2 = edge2mat(outward, num_node)
#     A3 = k_adjacency(A1, 2)
#     A4 = k_adjacency(A2, 2)
#     A1 = normalize_digraph(A1)
#     A2 = normalize_digraph(A2)
#     A3 = normalize_digraph(A3)
#     A4 = normalize_digraph(A4)
#     A = np.stack((I, A1, A2, A3, A4))
#     return A


# def get_uniform_graph(num_node, self_link, neighbor):
#     # 生成一个包含自环和邻居的二维邻接矩阵。
#     A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
#     return A


def edge2mat(link, num_node):
    # 将边列表转换为邻接矩阵，边列表中每个元素(i, j)表示从节点i到节点j有一条边，返回的邻接矩阵A[i][j]为1表示有边，为0表示没有边。
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    # 对于有向图，将邻接矩阵进行归一化，得到新的邻接矩阵。对于邻接矩阵A，新的邻接矩阵AD[i][j] = A[i][j] / sum(A[k][j])，其中k是所有出度为j的节点。
    """""
    这段代码实现了对有向图的邻接矩阵进行归一化操作，得到新的邻接矩阵。归一化是指将原矩阵的每一行元素除以该行元素之和，从而保证新矩阵每一行元素之和为1，方便后续处理。

    具体地，这里的函数 normalize_digraph(A) 的输入参数 A 是原始的邻接矩阵，其中 A[i][j] 表示从节点 i 到节点 j 是否存在边，若存在则为 1，否则为 0。
    函数首先计算了每个节点的出度，即从该节点出发的边的条数之和，存储在 Dl 中。
    然后，遍历每个节点，如果该节点的出度大于 0，则将 Dn[i][i] 赋值为 1/Dl[i]，否则为 0。这样可以得到一个对角矩阵 Dn，使得每个节点的出度都被归一化为 1。

    接着，函数将原邻接矩阵 A 与 Dn 相乘，得到新的邻接矩阵 AD。对于 AD[i][j]，它表示从节点 i 到节点 j 的权重，即原邻接矩阵 A 中 i 到 j 的边的权重除以所有出度为 j 的节点的出度之和。
    这样，对于每个节点 j，所有从其他节点到它的边的权重之和都被归一化为 1，方便后续的计算和处理。

    最后，函数返回新的邻接矩阵 AD。

    """
    Dl = np.sum(A, 0) # 指 每一列的和 ，计算每个节点的出度
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0: 
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn) # np.dot 指 矩阵乘法
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    """
    这是一个用于生成空间图的函数，它接受4个参数：节点数量(num_node)、自环链接(self_link)、入向链接(inward)和出向链接(outward)。
    首先，将自环链接转换为一个邻接矩阵I，然后将入向链接和出向链接分别转换为邻接矩阵In和Out，并进行归一化处理(normalize_digraph)。
    接下来，将I、In和Out这三个邻接矩阵沿着新的维度进行堆叠，形成一个三维数组A。
    最后，函数返回这个三维数组A，它代表了生成的空间图。

    In和Out是用来表示节点之间有方向性的边的邻接矩阵。相比于只有自连接的邻接矩阵I，它们提供了更加丰富的图信息，能够更好地表示节点之间的关系。
    例如，在社交网络中，如果只考虑用户之间的关注关系，那么每个用户之间的关系都是有方向性的，有些用户之间可能只有单向的关注关系，因此需要使用In和Out来区分不同的关系。
    """
    I = edge2mat(self_link, num_node) # [num_node, num_node] e.g. (19, 19)
    In = normalize_digraph(edge2mat(inward, num_node))  # [num_node, num_node] e.g. (19, 19)
    Out = normalize_digraph(edge2mat(outward, num_node)) # [num_node, num_node] e.g. (19, 19)
    A = np.stack((I, In, Out)) # [3, num_node, num_node] e.g. (3, 19, 19)  3代表特征维度？
    print('[get_spatial_graph] I:', I.shape, 'In:', In.shape, 'Out:', Out.shape, 'A:', A.shape)
    return A

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
