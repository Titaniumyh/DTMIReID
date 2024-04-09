import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class GraphGonvLayer(nn.Module):

    def __init__(self, in_dim, out_dim, topk):
        super(GraphGonvLayer, self).__init__()

        # parameters
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.topk = topk
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))

        self.reset_parameters()
        self.out = 0

        self.attn_linear = nn.Linear(in_dim, topk)

        # layers for feature
        self.fc_original_feature = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_merged_feature = nn.Linear(in_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        # learn adj
        # adj = self.learn_adj(inputs)
        adj = self.learn_adj2(inputs)
        # (1, 64, 64)
        merged_inputs = torch.matmul(adj, inputs)# (1, 64, 768)

        outputs1 = self.fc_merged_feature(merged_inputs)
        outputs2 = self.fc_original_feature(inputs)
        outputs = self.relu(outputs1) + outputs2
        #print(outputs.shape)
        return outputs

    def sum_scaler(self, w):
        s = torch.sum(w, dim=2)
        s = s.unsqueeze(2).repeat(1, 1, w.size(2))
        w = w / s
        return w

    def minmax_scaler(self, w):
        w_min, _ = torch.min(w, dim=2)
        w_max, _ = torch.max(w, dim=2)
        max_min = w_max - w_min
        max_min = max_min.unsqueeze(2).repeat(1, 1, w.size(2))
        w_min = w_min.unsqueeze(2).repeat(1, 1, w.size(2))
        w = (w - w_min) / max_min
        return w

    def stand_scaler(self, w):
        w_mean = torch.mean(w, dim=2)
        w_var = torch.var(w, dim=2)
        w_mean = w_mean.unsqueeze(2).repeat(1, 1, w.size(2))
        w_var = w_var.unsqueeze(2).repeat(1, 1, w.size(2))
        w = (w - w_mean) / w_var
        return w


    def learn_adj(self, x):
        # (1, 64, 768)
        w = torch.bmm(x / torch.norm(x, p=2, dim=2, keepdim=True),
                     (x / torch.norm(x, p=2, dim=2, keepdim=True)).transpose(1, 2))
        # _, rns_indices = torch.topk(torch.bmm(x / torch.norm(x, p=2, dim=2, keepdim=True),
        #                                       (x / torch.norm(x, p=2, dim=2, keepdim=True)).transpose(1, 2)), self.topk,
        #                             dim=2)
        _, rns_indices = torch.topk(w, self.topk, dim=2)
        # w = torch.matmul(x, x.permute(0, 2, 1))
        # w = self.sum_scaler(w)
        w = self.minmax_scaler(w)
        # w = self.stand_scaler(w)
        mask = torch.zeros_like(w).scatter_(2, rns_indices, torch.ones_like(rns_indices, dtype=w.dtype))
        mask = mask * mask.transpose(1, 2)


        if 'cuda' in str(x.device):
            mask = mask.cuda()
        else:
            mask = mask.cpu()

        w = w * mask + -1e9 * (1 - mask)
        w = F.softmax(w, dim=2)
        return w

    def learn_adj2(self, x):
        # (1, 64, 768)
        w = torch.bmm(x / torch.norm(x, p=2, dim=2, keepdim=True),
                     (x / torch.norm(x, p=2, dim=2, keepdim=True)).transpose(1, 2))
        # _, rns_indices = torch.topk(torch.bmm(x / torch.norm(x, p=2, dim=2, keepdim=True),
        #                                       (x / torch.norm(x, p=2, dim=2, keepdim=True)).transpose(1, 2)), self.topk,
        #                             dim=2)
        _, rns_indices = torch.topk(w, self.topk, dim=2)
        # w = torch.matmul(x, x.permute(0, 2, 1))
        # w = self.sum_scaler(w)
        # w = self.minmax_scaler(w)
        # w = self.stand_scaler(w)
        mask = torch.zeros_like(w).scatter_(2, rns_indices, torch.ones_like(rns_indices, dtype=w.dtype))
        mask = mask * mask.transpose(1, 2)


        if 'cuda' in str(x.device):
            mask = mask.cuda()
        else:
            mask = mask.cpu()

        attn = self.attn_linear(x)
        attn = torch.zeros_like(w).scatter_(2, rns_indices, attn)
        attn = attn.to(x.device)
        attn = attn * mask + -1e9 * (1 - mask)
        # w = w * mask + -1e9 * (1 - mask)
        attn = F.softmax(attn, dim=2)
        return attn

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_dim) + ' -> ' \
               + str(self.out_dim) + ')'

class GCN(nn.Module):
    def __init__(self, f_dim, topk, layer_num):
        super(GCN, self).__init__()
        self.layer = GraphGonvLayer(f_dim, f_dim, topk)
        self.layers = nn.ModuleList([copy.deepcopy(self.layer) for i in range(layer_num)])

    def forward(self, x):
        # x(1, 64, 768)
        for layer in self.layers:
            x = layer(x)
        return x