"""GSAT with a four-layer attention-weighted DrugOOD GIN."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing, global_add_pool

class AttGINConv(MessagePassing):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__(aggr="add")
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim), nn.BatchNorm1d(2*hidden_dim),
                                 nn.ReLU(), nn.Linear(2*hidden_dim, hidden_dim))
    def forward(self, x, edge_index, edge_attr, edge_attention=None):
        out = self.propagate(edge_index, x=x, edge_attr=self.edge_encoder(edge_attr.float()), edge_attention=edge_attention)
        return self.mlp((1+self.eps)*x + out)
    def message(self, x_j, edge_attr, edge_attention):
        msg = F.relu(x_j + edge_attr)
        return msg if edge_attention is None else msg * edge_attention

class GSATModel(nn.Module):
    def __init__(self, node_dim=39, edge_dim=10, hidden_dim=128, layers=4, dropout=0.1):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([AttGINConv(hidden_dim, edge_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(layers)])
        self.extractor = nn.Sequential(nn.Linear(2*hidden_dim, 4*hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                       nn.Linear(4*hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                       nn.Linear(hidden_dim, 1))
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout
    def encode(self, batch, edge_attention=None):
        x = self.node_encoder(batch.x.float())
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = norm(conv(x, batch.edge_index, batch.edge_attr, edge_attention))
            if i+1 < len(self.convs): x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x
    def attention_logits(self, batch):
        emb = self.encode(batch)
        row, col = batch.edge_index
        return self.extractor(torch.cat((emb[row], emb[col]), -1))
    @staticmethod
    def sample(logits, training):
        if not training: return logits.sigmoid()
        noise = torch.empty_like(logits).uniform_(1e-10, 1-1e-10)
        return (logits + noise.log() - (1-noise).log()).sigmoid()
    @staticmethod
    def symmetrize(attention, edge_index, num_nodes):
        if attention.numel() == 0:
            return attention
        row, col = edge_index
        keys = torch.minimum(row, col) * num_nodes + torch.maximum(row, col)
        _, inverse = torch.unique(keys, return_inverse=True)
        sums = attention.new_zeros((int(inverse.max())+1, 1)).index_add_(0, inverse, attention)
        counts = attention.new_zeros((sums.shape[0], 1)).index_add_(0, inverse, torch.ones_like(attention))
        return (sums/counts.clamp_min(1))[inverse]
    def forward_gsat(self, batch, training):
        raw = self.sample(self.attention_logits(batch), training)
        edge_att = self.symmetrize(raw, batch.edge_index, batch.x.shape[0])
        rep = global_add_pool(self.encode(batch, edge_att), batch.batch)
        return self.classifier(rep), raw, edge_att
    def forward_erm(self, batch):
        return self.classifier(global_add_pool(self.encode(batch), batch.batch))
