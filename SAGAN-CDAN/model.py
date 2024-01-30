from audioop import bias
from bisect import bisect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  global_add_pool, global_mean_pool
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import  softmax
from torch_scatter import scatter
from torch_geometric.utils import degree
def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    # n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        # loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = nn.NLLLoss()(class_output.squeeze(), label.long())
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return 1, loss

class GlobalAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GATConv(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)
        scores = softmax(x_conv, batch, dim=0)
        gx = global_add_pool(x * scores, batch)

        return gx


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

        glorot(self.a)

        self.lin_block = LinearBlock(n_feats)

    def forward(self, data):

        edge_index = data.edge_index

        edge_u = self.lin_u(data.x)
        edge_v = self.lin_v(data.x)
        edge_uv = self.lin_edge(data.edge_attr)
        edge_attr = (edge_u[edge_index[0]] + edge_v[edge_index[1]] + edge_uv) / 3
        out = edge_attr
        
        out_list = []
        gout_list = []
        for n in range(self.n_iter):
            out = scatter(out[data.line_graph_edge_index[0]] , data.line_graph_edge_index[1], dim_size=edge_attr.size(0), dim=0, reduce='add')
            out = edge_attr + out
            gout = self.att(out, data.line_graph_edge_index, data.edge_index_batch)
            out_list.append(out)
            gout_list.append(F.tanh((self.lin_gout(gout))))

        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)
        scores = (gout_all * self.a).sum(1, keepdim=True) + self.a_bias
        scores = torch.softmax(scores, dim=-1)
        scores = scores.repeat_interleave(degree(data.edge_index_batch, dtype=data.edge_index_batch.dtype), dim=0)

        out = (out_all * scores).sum(-1)

        x = data.x + scatter(out , edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')
        x = self.lin_block(x)

        return x

class DrugEncoder1(torch.nn.Module):
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
class DrugEncoder2(torch.nn.Module):
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
    def __init__(self, in_dim1,in_dim2, edge_in_dim, hidden_dim=64, n_iter=10):
        super(SA_DDI, self).__init__()

        self.drug_encoder1 = DrugEncoder1(in_dim1, edge_in_dim, hidden_dim, n_iter=n_iter)
        self.drug_encoder2 = DrugEncoder2(in_dim2, edge_in_dim, hidden_dim, n_iter=n_iter)
        self.h_gpool = GlobalAttentionPool(hidden_dim)
        self.t_gpool = GlobalAttentionPool(hidden_dim)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.rmodule = nn.Embedding(963, hidden_dim)

        self.w_j = nn.Linear(hidden_dim, hidden_dim)
        self.w_i = nn.Linear(hidden_dim, hidden_dim)

        self.prj_j = nn.Linear(hidden_dim, hidden_dim)
        self.prj_i = nn.Linear(hidden_dim, hidden_dim)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
    def forward(self, triples):
        a=len(triples)
        if len(triples)==6:
            h_data_s, t_data_s, rels_s,h_data_t, t_data_t, rels_t = triples
            #%% sdata
            x_h_s = self.drug_encoder1(h_data_s)
            x_t_s = self.drug_encoder1(t_data_s)

            g_h_s = self.h_gpool(x_h_s, h_data_s.edge_index, h_data_s.batch)
            g_t_s = self.t_gpool(x_t_s, t_data_s.edge_index, t_data_s.batch)

            g_h_align_s = g_h_s.repeat_interleave(degree(t_data_s.batch, dtype=t_data_s.batch.dtype), dim=0)
            g_t_align_s = g_t_s.repeat_interleave(degree(h_data_s.batch, dtype=h_data_s.batch.dtype), dim=0)

            h_scores_s = (self.w_i(x_h_s) * self.prj_i(g_t_align_s)).sum(-1)
            h_scores_s = softmax(h_scores_s, h_data_s.batch, dim=0)

            t_scores_s = (self.w_j(x_t_s) * self.prj_j(g_h_align_s)).sum(-1)
            t_scores_s = softmax(t_scores_s, t_data_s.batch, dim=0)

            h_final_s = global_add_pool(x_h_s * g_t_align_s * h_scores_s.unsqueeze(-1), h_data_s.batch)
            t_final_s = global_add_pool(x_t_s * g_h_align_s * t_scores_s.unsqueeze(-1), t_data_s.batch)

            pair_s = torch.cat([h_final_s, t_final_s], dim=-1)
            rfeat_s = self.rmodule(rels_s)
            logit_s = (self.lin(pair_s) * rfeat_s).sum(-1)
            logit_s = logit_s * pair_s.t()
            x_s = self.relu1(self.bn1(self.fc1(logit_s.t())))
            x_s = self.relu2(self.bn2(self.fc2(x_s)))#(512,128)
            x_s = self.fc3(x_s)
            # %% tdata
            x_h_t = self.drug_encoder2(h_data_t)
            x_t_t = self.drug_encoder2(t_data_t)

            g_h_t = self.h_gpool(x_h_t, h_data_t.edge_index, h_data_t.batch)
            g_t_t = self.t_gpool(x_t_t, t_data_t.edge_index, t_data_t.batch)

            g_h_align_t = g_h_t.repeat_interleave(degree(t_data_t.batch, dtype=t_data_t.batch.dtype), dim=0)
            g_t_align_t = g_t_t.repeat_interleave(degree(h_data_t.batch, dtype=h_data_t.batch.dtype), dim=0)

            h_scores_t = (self.w_i(x_h_t) * self.prj_i(g_t_align_t)).sum(-1)
            h_scores_t = softmax(h_scores_t, h_data_t.batch, dim=0)

            t_scores_t = (self.w_j(x_t_t) * self.prj_j(g_h_align_t)).sum(-1)
            t_scores_t = softmax(t_scores_t, t_data_t.batch, dim=0)

            h_final_t = global_add_pool(x_h_t * g_t_align_t * h_scores_t.unsqueeze(-1), h_data_t.batch)
            t_final_t = global_add_pool(x_t_t * g_h_align_t * t_scores_t.unsqueeze(-1), t_data_t.batch)

            pair_t = torch.cat([h_final_t, t_final_t], dim=-1)
            rfeat_t = self.rmodule(rels_t)
            logit_t = (self.lin(pair_t) * rfeat_t).sum(-1)
            logit_t1 = logit_t * pair_t.t()
            x_t = self.relu1(self.bn1(self.fc1(logit_t1.t())))
            x_t = self.relu2(self.bn2(self.fc2(x_t)))  # (512,128)
            x_t = self.fc3(x_t)
        else:
            a=0
        return logit_t,x_s,x_t

