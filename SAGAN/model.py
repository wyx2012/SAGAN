import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.utils import degree
from collections import defaultdict
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

class GlobalAttentionPool(nn.Module):
    '''
    This is the topology-aware global pooling mentioned in the paper.
    '''

    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GATConv(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)
        scores = softmax(x_conv, batch, dim=0)
        gx = global_add_pool(x * scores, batch)

        return gx

class GlobalAttentionPoolh(nn.Module):
    '''
    This is the topology-aware global pooling mentioned in the paper.
    '''

    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GATConv(hidden_dim, 30)
        self.l1 = nn.Linear(30, 1)
        self.l2 = nn.Linear(30, 64)
    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)#x_conv【13222，1】
        #在这里对x_conv进行聚类
        _, indices = torch.unique(batch, return_counts=True)
        x_h_0 = torch.split(x_conv, indices.tolist())
        x_h_n = []
        i=0
        batch_n=[]
        for value in x_h_0:
            value1 = (value > 0).float()
            value1 = value1.squeeze(dim=1)
            bandwidth = 0.2
            # bandwidth = estimate_bandwidth(value1, quantile=0.5, n_samples=2)
            # 创建MeanShift实例并进行拟合
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(value1.cpu())
            labels = ms.labels_
            _, s = np.unique(labels, return_index=True)
            value1_latter = torch.index_select(value, dim=0, index=torch.from_numpy(s).cuda(0))
            # cluster_centers = ms.cluster_centers_
            x_h_n.append(value1_latter)  # list512,[7,64]
            batch_n.extend([i] * value1_latter.shape[0])
            i=i+1
        x_h_n = torch.cat(x_h_n, dim=0)  # 10850.64
        batch_n = torch.tensor(batch_n)
        x_h_1 = self.l1(x_h_n)
        x_h_64 = self.l2(x_h_n)
        scores = softmax(x_h_1,batch_n.cuda(0), dim=0)
        gx = global_add_pool(x_h_64 * scores, batch_n.cuda(0))
        return gx
class GlobalAttentionPoolt(nn.Module):
    '''
    This is the topology-aware global pooling mentioned in the paper.
    '''

    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GATConv(hidden_dim, 30)
        self.l1 = nn.Linear(30, 1)
        self.l2 = nn.Linear(30, 64)
    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)  # x_conv【13222，1】
        # 在这里对x_conv进行聚类
        _, indices = torch.unique(batch, return_counts=True)
        x_h_0 = torch.split(x_conv, indices.tolist())
        x_h_n = []
        i = 0
        batch_n = []
        for value in x_h_0:
            value1 = (value > 0).float()
            value1 = value1.squeeze(dim=1)
            bandwidth = 0.2
            # bandwidth = estimate_bandwidth(value1, quantile=0.5, n_samples=2)
            # 创建MeanShift实例并进行拟合
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(value1.cpu())
            labels = ms.labels_
            _, s = np.unique(labels, return_index=True)
            value1_latter = torch.index_select(value, dim=0, index=torch.from_numpy(s).cuda(0))
            # cluster_centers = ms.cluster_centers_
            x_h_n.append(value1_latter)  # list512,[7,64]
            batch_n.extend([i] * value1_latter.shape[0])
            i = i + 1
        x_h_n = torch.cat(x_h_n, dim=0)  # 10850.64
        batch_n = torch.tensor(batch_n)
        x_h_1 = self.l1(x_h_n)
        x_h_64 = self.l2(x_h_n)
        scores = softmax(x_h_1, batch_n.cuda(0), dim=0)
        gx = global_add_pool(x_h_64 * scores, batch_n.cuda(0))
        return gx

class DMPNN(nn.Module):
    def __init__(self, edge_dim, n_feats, n_iter):
        super().__init__()
        self.n_iter = n_iter

        self.lin_u = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_v = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_edge = nn.Linear(edge_dim, n_feats, bias=False)

        self.att = GlobalAttentionPool(n_feats)
        self.a = nn.Parameter(torch.zeros(1, n_feats, n_iter))
        self.lin_gout = nn.Linear(n_feats, n_feats)
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_iter))
        self.sigmoid = torch.nn.Sigmoid()
        glorot(self.a)

        self.lin_block = LinearBlock(n_feats)

    def forward(self, data):
        edge_index = data.edge_index
        # Recall that we have converted the node graph to the line graph,
        # so we should assign each bond a bond-level feature vector at the beginning (i.e., h_{ij}^{(0)}) in the paper).
        edge_u = self.lin_u(data.x)
        edge_v = self.lin_v(data.x)
        edge_uv = self.lin_edge(data.edge_attr)
        edge_attr = (edge_u[edge_index[0]] + edge_v[edge_index[1]] + edge_uv) / 3
        out = edge_attr

        # The codes below show the graph convolution and substructure attention.
        out_list = []
        gout_list = []
        for n in range(self.n_iter):
            # Lines 61 and 62 are the main steps of graph convolution.
            out = scatter(out[data.line_graph_edge_index[0]], data.line_graph_edge_index[1], dim_size=edge_attr.size(0),
                          dim=0, reduce='add')
            out = edge_attr + out
            # Equation (1) in the paper
            gout = self.att(out, data.line_graph_edge_index, data.edge_index_batch)
            out_list.append(out)
            gout_list.append(F.tanh((self.lin_gout(gout))))

        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)
        # Substructure attention, Equation (3)
        scores = (gout_all * self.a).sum(1, keepdim=True) + self.a_bias
        # Substructure attention, Equation (4),
        # Suppose batch_size=64 and iteraction_numbers=10.
        # Then the scores will have a shape of (64, 1, 10),
        # which means that each graph has 10 scores where the n-th score represents the importance of substructure with radius n.
        scores = torch.softmax(scores, dim=-1)
        # We should spread each score to every line in the line graph.
        scores = scores.repeat_interleave(degree(data.edge_index_batch, dtype=data.edge_index_batch.dtype), dim=0)
        # Weighted sum of bond-level hidden features across all steps, Equation (5).
        out = (out_all * scores).sum(-1)
        # Return to node-level hidden features, Equations (6)-(7).
        x = data.x + scatter(out, edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')
        x = self.lin_block(x)

        return x


class LinearBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.snd_n_feats = 6 * n_feats
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )
        self.lin5 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, n_feats)
        )

    def forward(self, x):
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2
        x = self.lin5(x)

        return x


class DrugEncoder(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.lin0 = nn.Linear(in_dim, hidden_dim)
        self.line_graph = DMPNN(edge_in_dim, hidden_dim, n_iter)

    def forward(self, data):
        data.x = self.mlp(data.x)
        x = self.line_graph(data)

        return x


class SA_DDI(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10):
        super(SA_DDI, self).__init__()
        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_iter=n_iter)
        self.h_gpool = GlobalAttentionPoolh(hidden_dim)
        self.t_gpool = GlobalAttentionPoolt(hidden_dim)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.rmodule = nn.Embedding(86, hidden_dim)

        self.w_j = nn.Linear(hidden_dim, hidden_dim)
        self.w_i = nn.Linear(hidden_dim, hidden_dim)

        self.prj_j = nn.Linear(hidden_dim, hidden_dim)
        self.prj_i = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, triples):
        h_data, t_data, rels = triples

        x_h = self.drug_encoder(h_data)
        x_t = self.drug_encoder(t_data)
        #聚类开始
        _, indices = torch.unique(h_data.batch, return_counts=True)
        x_h_0 = torch.split(x_h, indices.tolist())
        x_h_n =[]
        for value in x_h_0:
            value1 = (value > 0).float()
            bandwidth = 0.2
            # bandwidth = estimate_bandwidth(value1, quantile=0.5, n_samples=2)
            # 创建MeanShift实例并进行拟合
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(value1.cpu())
            labels = ms.labels_
            _, s = np.unique(labels, return_index=True)
            value1_latter = torch.index_select(value,dim=0,index=torch.from_numpy(s).cuda(0))
            # cluster_centers = ms.cluster_centers_
            x_h_n.append(value1_latter)#list512,[7,64]
        x_h_n = torch.cat(x_h_n, dim=0)#10850.64
        #聚类结束
        # Start of SSIM
        print('Start of SSIM')
        # TAGP, Equation (8)
        g_h = self.h_gpool(x_h, h_data.edge_index, h_data.batch)
        g_t = self.t_gpool(x_t, t_data.edge_index, t_data.batch)

        g_h_align = g_h.repeat_interleave(degree(t_data.batch, dtype=t_data.batch.dtype), dim=0)
        g_t_align = g_t.repeat_interleave(degree(h_data.batch, dtype=h_data.batch.dtype), dim=0)

        # Equation (10)
        h_scores = (self.w_i(x_h) * self.prj_i(g_t_align)).sum(-1)
        h_scores = softmax(h_scores, h_data.batch, dim=0)
        # Equation (10)
        t_scores = (self.w_j(x_t) * self.prj_j(g_h_align)).sum(-1)
        t_scores = softmax(t_scores, t_data.batch, dim=0)
        # Equation (11)
        h_final = global_add_pool(x_h * g_t_align * h_scores.unsqueeze(-1), h_data.batch)
        t_final = global_add_pool(x_t * g_h_align * t_scores.unsqueeze(-1), t_data.batch)
        # End of SSIM

        pair = torch.cat([h_final, t_final], dim=-1)
        #和这个pair相加或相连就可以（512.128）
        rfeat = self.rmodule(rels)
        logit = (self.lin(pair) * rfeat).sum(-1)

        return logit