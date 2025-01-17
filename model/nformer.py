import copy
import json
import math
import re
import collections

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {
    'relu': nn.ReLU,
    'swish': swish,
    'gelu': gelu
}


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_head, dropout, scale=False):
        super(Attention, self).__init__()
        n_state = nx
        assert n_state % n_head == 0
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)

        self.resid_dropout = nn.Dropout(dropout)

    def _attn(self, q, k, v, num_landmark, rns_indices):
        data_length = q.shape[2]
        landmark = torch.Tensor(random.sample(range(data_length), num_landmark)).long()

        sq = q[:, :, landmark, :].contiguous()
        sk = k[:, :, :, landmark].contiguous()

        w1 = torch.matmul(q, sk)
        w2 = torch.matmul(sq, k)
        w = torch.matmul(w1, w2)
        w = torch.matmul(q, k)

        if self.scale:
            w = w / math.sqrt(v.size(-1))
        return self.rns(w, v, rns_indices)

    def nodes_mask(self):
        bs = 64
        np = 7
        m = torch.zeros((bs * np, bs * np))
        i = 0
        for j in range(bs):
            m[i:i + 7, i:i + 7] = 1
            i = i + 7
        m1 = torch.ones((bs * np, bs * np))
        m1 = m1 - m
        m = m * 0.3
        m1 = m1 * 1
        m = m + m1
        z = torch.eye(bs*np)
        m = m * (1 - z) + z
        return m.cuda()


    def rns(self, w, v, rns_indices):
        bs, hn, dl, _ = w.shape
        rns_indices = rns_indices.unsqueeze(1).repeat(1, hn, 1, 1)
        mask = torch.zeros_like(w).scatter_(3, rns_indices, torch.ones_like(rns_indices, dtype=w.dtype))
        mask = mask * mask.transpose(2, 3)
        if 'cuda' in str(w.device):
            mask = mask.cuda()
        else:
            mask = mask.cpu()
        if self.training:
            w = w * mask + -1e9 * (1 - mask)
            w = F.softmax(w, dim=3)
            # w = w * self.nodes_mask()
            a_v = torch.matmul(w, v)
        else:
            w = (w * mask).reshape(bs * hn, dl, dl).to_sparse()
            w = torch.sparse.softmax(w, 2)
            v = v.reshape(bs * hn, dl, -1)
            a_v = torch.bmm(w, v).reshape(bs, hn, dl, -1)
        return a_v

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, num_landmark, rns_indices):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        mask = None
        a = self._attn(query, key, value, num_landmark, rns_indices)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(nn.Module):
    def __init__(self, n_state, dim, dropout):
        super(MLP, self).__init__()
        nx = dim
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS['gelu']
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, dim, n_head, dropout, scale=False):
        super(Block, self).__init__()
        nx = dim
        self.attn = Attention(nx, n_head, dropout, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, dim, dropout)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x, num_landmark, rns_indices):
        a = self.attn(x, num_landmark, rns_indices)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


# class NFormer(nn.Module):
#     """ NFormer model """
#
#     def __init__(self, dim, n_head, layers, dropout, topk, num_landmark, num_classes=751):
#         super(NFormer, self).__init__()
#         self.num_classes = num_classes
#         self.norm = nn.LayerNorm(dim)
#
#         block = Block(dim, n_head, dropout, scale=True)
#         self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(layers)])
#
#         # self.bottleneck = nn.BatchNorm1d(dim)
#         # self.bottleneck.bias.requires_grad_(False)  # no shift
#         # self.bottleneck.apply(weights_init_kaiming)
#         #
#         # self.classifier = nn.Linear(dim, self.num_classes, bias=False)
#         # self.classifier.apply(weights_init_classifier)
#         self.topk = topk
#         self.num_landmark = num_landmark
#
#     def forward(self, x):
#         x = self.norm(x)
#         _, rns_indices = torch.topk(torch.bmm(x / torch.norm(x, p=2, dim=2, keepdim=True),
#                                               (x / torch.norm(x, p=2, dim=2, keepdim=True)).transpose(1, 2)), self.topk,
#                                     dim=2)
#         for block in self.h:
#             x = block(x, self.num_landmark, rns_indices)
#
#         # bs, dl, d = x.shape
#         # x = x.reshape(bs * dl, d)
#         # feat = self.bottleneck(x)
#         # cls_score = self.classifier(feat)
#         return x
#         # x = x.reshape(bs, dl, d)
#         # feat = feat.reshape(bs, dl, d)
#         # cls_score = cls_score.reshape(bs, dl, -1)
#
#         # if self.training:
#         #     return cls_score, x
#         # else:
#         #     return feat
class NFormer(nn.Module):
    """ NFormer model """

    def __init__(self, dim, n_head, layers, dropout, topk, num_landmark, num_classes=751):
        super(NFormer, self).__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(dim)

        block = Block(dim, n_head, dropout, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(layers)])

        self.bottleneck = nn.BatchNorm1d(dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.topk = topk
        self.num_landmark = num_landmark

    def forward(self, x):
        x = self.norm(x)
        _, rns_indices = torch.topk(torch.bmm(x / torch.norm(x, p=2, dim=2, keepdim=True),
                                              (x / torch.norm(x, p=2, dim=2, keepdim=True)).transpose(1, 2)), self.topk,
                                    dim=2)
        for block in self.h:
            x = block(x, self.num_landmark, rns_indices)

        bs, dl, d = x.shape
        x = x.reshape(bs * dl, d)
        feat = self.bottleneck(x)
        cls_score = self.classifier(feat)
        # return x
        x = x.reshape(bs, dl, d)
        feat = feat.reshape(bs, dl, d)
        cls_score = cls_score.reshape(bs, dl, -1)
        return x, cls_score

        # if self.training:
        #     return cls_score, x
        # else:
        #     return feat

# class NFormerlayer(nn.Module):
#     """ NFormer model """
#
#     def __init__(self, dim, n_head, dropout, topk, num_landmark, num_classes=751):
#         super(NFormerlayer, self).__init__()
#         self.num_classes = num_classes
#         self.norm = nn.LayerNorm(dim)
#
#         self.block = Block(dim, n_head, dropout, scale=True)
#         # self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(layers)])
#
#         # self.bottleneck = nn.BatchNorm1d(dim)
#         # self.bottleneck.bias.requires_grad_(False)  # no shift
#         # self.bottleneck.apply(weights_init_kaiming)
#         #
#         # self.classifier = nn.Linear(dim, self.num_classes, bias=False)
#         # self.classifier.apply(weights_init_classifier)
#         self.topk = topk
#         self.num_landmark = num_landmark
#
#     def forward(self, x):
#         x = self.norm(x)
#         _, rns_indices = torch.topk(torch.bmm(x / torch.norm(x, p=2, dim=2, keepdim=True),
#                                               (x / torch.norm(x, p=2, dim=2, keepdim=True)).transpose(1, 2)), self.topk,
#                                     dim=2)
#         x = self.block(x, self.num_landmark, rns_indices)
#
#         return x
#
#         # if self.training:
#         #     return cls_score, x
#         # else:
#         #     return feat
# class NFormerMulti(nn.Module):
#     def __init__(self, dim, n_head, dropout, topk, num_landmark, num_classes=751):
#         super(NFormerMulti, self).__init__()
#         self.nformer_layer = NFormerlayer(dim, n_head, dropout, topk, num_landmark, num_classes=num_classes)
#         self.layers = nn.ModuleList([copy.deepcopy(self.nformer_layer) for _ in range(12)])
#     def forward(self, feats):
#         # feats[12]  (b,211,768)

if __name__ == '__main__':
    nf = NFormer(256, 2, 4, 0.1, 20, 5, num_classes=702)