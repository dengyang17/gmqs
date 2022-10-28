import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn
#from torch_geometric.nn import GCNConv, RGCNConv, GATConv

import models

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output





class RGCN(nn.Module):
    def __init__(self, config):# nfeat, nhid, nlayer, nrel, dropout
        super(RGCN, self).__init__()
        self.num_hops = config.num_hops
        self.rgcns = nn.ModuleList()
        for j in config.relations:
            gcns = nn.ModuleList()
            for i in range(config.num_hops):
                gcns.append(GraphConvolution(config.hidden_size, config.hidden_size))
            self.rgcns.append(gcns)
        self.dropout = config.dropout
        self.wo = nn.Linear(config.hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adjs, mask):
        adjs = adjs.permute(1,0,2,3)
        for i in range(self.num_hops-1):
            x = torch.stack([self.rgcns[j][i](x, adjs[j]) for j in range(len(self.rgcns))], dim=2)
            x = F.relu(torch.sum(x, dim=2))
            x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.relu(torch.sum(torch.stack([self.rgcns[j][-1](x, adjs[j]) for j in range(len(self.rgcns))], dim=2), dim=2))[:,1:]

        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return x, sent_scores





class GATLayer(Module):

    def __init__(self, in_features, out_features, att_dim, hidden_size):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.att_dim = att_dim
        self.leakyrelu = nn.LeakyReLU(1e-2)

        self.W = nn.Linear(in_features, att_dim)

        a_layers = [nn.Linear(2 * att_dim, hidden_size), 
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)]
        self.afcs = nn.Sequential(*a_layers)

    def forward(self, input, adj):
        B, N = adj.size(0), adj.size(1)
        dmask = adj.view(B, -1)  # (batch_size, n*n)

        h = self.W(input) # (B, N, D)
        a_input = torch.cat([h.repeat(1, 1, N).view(
                B, N*N, -1), h.repeat(1, N, 1)], dim=2)  # (B, N*N, 2*D)
        e = self.leakyrelu(self.afcs(a_input)).squeeze(2)  # (B, N*N)
        attention = F.softmax(mask_logits(e, dmask), dim=1)
        attention = attention.view(*adj.size())

        output = attention.bmm(h)
        return output

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)



class RGAT(nn.Module):
    def __init__(self, config):# nfeat, nhid, nclass, nlayer, nrel, att_dim, hidden_size, dropout
        super(RGAT, self).__init__()
        self.num_hops = config.num_hops
        self.rgats = nn.ModuleList()
        for j in config.relations:
            gats = nn.ModuleList()
            for i in range(config.num_hops):
                gats.append(GATLayer(config.hidden_size, config.hidden_size, config.hidden_size, config.hidden_size))
            self.rgats.append(gats)
        self.dropout = config.dropout
        self.wo = nn.Linear(config.hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adjs, mask):
        adjs = adjs.permute(1,0,2,3)
        for i in range(self.num_hops-1):
            x = torch.stack([self.rgats[j][i](x, adjs[j]) for j in range(len(self.rgats))], dim=2)
            x = F.relu(torch.sum(x, dim=2))
            x = F.dropout(x, self.dropout, training=self.training)
        #return F.relu(torch.sum(torch.stack([self.rgats[j][-1](x, adjs[j]) for j in range(len(self.rgats))], dim=2), dim=2))

        x = F.relu(torch.sum(torch.stack([self.rgats[j][-1](x, adjs[j]) for j in range(len(self.rgats))], dim=2), dim=2))[:,1:]

        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return x, sent_scores




class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        x = x[:,1:]
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return x, sent_scores



class TransformerInterEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerInterEncoderLayer, self).__init__()

        self.self_attn = models.Multihead_Attention(
            d_model, heads, dropout=dropout)
        self.feed_forward = models.PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerInterEncoder(nn.Module):
    def __init__(self, config):#d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerInterEncoder, self).__init__()
        self.d_model = config.hidden_size
        self.num_inter_layers = config.num_hops
        self.pos_emb = models.PositionalEncoding(config.dropout, config.hidden_size)
        self.transformer_inter = nn.ModuleList(
            [TransformerInterEncoderLayer(config.hidden_size, config.heads, config.d_ff, config.dropout)
             for _ in range(config.num_hops)])
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.wo = nn.Linear(config.hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)[:,1:]
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return x, sent_scores







'''
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(GCN, self).__init__()

        self.gcns = []
        for i in nlayer:
            if i == 0:
                self.gcns.append(GraphConvolution(nfeat, nhid))
            else:
                self.gcns.append(GraphConvolution(nhid, nhid))
        self.dropout = dropout

    def forward(self, x, adj):
        for gcn in self.gcns[:-1]:
            x = F.relu(gcn(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        return F.relu(gcn[-1](x, adj))

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nlayer, att_dim, hidden_size, dropout):
        super(GAT, self).__init__()
        self.gats = []
        for i in nlayer:
            if i == 0:
                self.gats.append(GATLayer(nfeat, nhid, att_dim, hidden_size))
            else:
                self.gats.append(GATLayer(nhid, nhid, att_dim, hidden_size))
        self.dropout = dropout

    def forward(self, x, adj):
        for gat in self.gats[:-1]:
            x = F.relu(gat(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        return F.relu(gat[-1](x, adj))
'''