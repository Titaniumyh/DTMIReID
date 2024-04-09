import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


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


class BatchNorm(nn.Module):
    def __init__(self, dim):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.bn.bias.requires_grad_(False)
        self.bn.apply(weights_init_kaiming)

    def forward(self, x):
        return self.bn(x)


class Classifier(nn.Module):
    def __init__(self, dim, class_num):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
    def forward(self, x):
        return self.classifier(x)

class CrossAttn(nn.Module):
    def __init__(self, dim, nh):
        super(CrossAttn, self).__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)
        self.l3 = nn.Linear(dim, dim)
        self.nh = nh

    def forward(self, q, k, v, rns_indices):
        np, bs, dim = q.size()
        mnp = k.size(0)
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        q = self.l1(q).reshape(bs, np, self.nh, -1).permute(0, 2, 1, 3)
        k = self.l2(k).reshape(bs, mnp, self.nh, -1).permute(0, 2, 3, 1)
        v = self.l3(v).reshape(bs, mnp, self.nh, -1).permute(0, 2, 1, 3)
        w = torch.matmul(q, k)
        w = w / math.sqrt(v.size(-1))
        rns_indices = rns_indices.unsqueeze(1).repeat(1, self.nh, 1, 1)
        mask = torch.zeros_like(w).scatter_(3, rns_indices, torch.ones_like(rns_indices, dtype=w.dtype))
        if 'cuda' in str(w.device):
            mask = mask.cuda()
        else:
            mask = mask.cpu()
        w = w * mask + -1e9 * (1 - mask)
        w = F.softmax(w, dim=3)
        v = torch.matmul(w, v).permute(2, 0, 1, 3).reshape(np, bs, -1)
        return v

class ReDetrDecoder(nn.Module):
    def __init__(self, dim, num_patch, num_layers, nheads, num_classes, topk, return_multi=False, cross_first=True, dim_feedforward=2048, dropout=0.1):
        super(ReDetrDecoder, self).__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.return_multi = return_multi
        self.cls_tgt = nn.Linear(dim, num_patch)
        self.norm0 = nn.LayerNorm(dim)
        self.layer = DetrDecoderLayer(dim, nheads, topk, dim_feedforward, dropout, cross_first)
        self.layers = nn.ModuleList(copy.deepcopy(self.layer) for _ in range(num_layers))

        self.bn = BatchNorm(dim)
        self.bns = nn.ModuleList(copy.deepcopy(self.bn) for _ in range(num_layers))

        self.classifier = Classifier(dim, num_classes)
        self.classifiers = nn.ModuleList(copy.deepcopy(self.classifier) for _ in range(num_layers))

    def forward_head(self, tgt):
        return self.cls_tgt(tgt)

    def forward(self, tgt, memory, query_pos=None, key_pos=None):
        # cls_out = []
        bs, np, d = tgt.shape  # batch_size, num_patch, dim
        tgt = self.norm0(tgt)
        # cls_pred = self.forward_head(tgt)
        # cls_out.append(cls_pred)
        if self.return_multi:
            output = []
            cls_scores = []
            output.append(tgt)
            cls = tgt.reshape(np * bs, d)
            cls = self.bn(cls)
            cls = self.classifier(cls)
            cls_scores.append(cls.reshape(bs, np, -1))

        tgt = tgt.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        key_pos = key_pos.permute(1, 0, 2)
        i = 0
        for layer in self.layers:
            tgt = layer(tgt, memory, query_pos, key_pos)
            if self.return_multi:
                cls = tgt.reshape(np * bs, d)
                cls = self.bns[i](cls)
                cls = self.classifiers[i](cls)
                cls = cls.reshape(np, bs, self.num_classes)
                cls_scores.append(cls.permute(1, 0, 2))
                output.append(tgt.permute(1, 0, 2))
                i = i + 1

        if self.return_multi:
            return output, cls_scores
        else:
            cls = self.bn(tgt.reshape(np * bs, d))
            cls = self.classifier(cls)
            cls = cls.reshape(np, bs, self.num_classes)
            return tgt.permute(1, 0, 2), cls.permute(1, 0, 2)


class DetrDecoderLayer(nn.Module):
    def __init__(self, dim, nheads, topk, dim_feedforward=2048, dropout=0.1, cross_frist=True):
        super(DetrDecoderLayer, self).__init__()
        self.cross_first = cross_frist
        # self.cross_attn = nn.MultiheadAttention(dim, nheads, dropout)
        self.cross_attn = CrossAttn(dim, nheads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)

        self.self_attn = nn.MultiheadAttention(dim, nheads, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)

        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.activation = F.gelu
        self.topk = topk

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_cross_first(self, tgt, memory, rns_indices, query_pos=None, key_pos=None):
        tgt2 = self.cross_attn(q=self.with_pos_embed(tgt, query_pos),
                               k=self.with_pos_embed(memory, key_pos),
                               v=memory,
                               rns_indices=rns_indices)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # self_attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward_self_first(self, tgt, memory, query_pos=None, key_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.cross_attn(query=self.with_pos_embed(tgt, query_pos),
                               key=self.with_pos_embed(memory, key_pos),
                               value=memory)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory, query_pos=None, key_pos=None):
        x = tgt.permute(1, 0, 2)
        m = memory.permute(1, 0, 2)
        _, rns_indices = torch.topk(torch.bmm(x / torch.norm(x, p=2, dim=2, keepdim=True),
                                              (m / torch.norm(m, p=2, dim=2, keepdim=True)).transpose(1, 2)), self.topk,
                                    dim=2)
        if self.cross_first:
            tgt = self.forward_cross_first(tgt, memory, rns_indices, query_pos=query_pos, key_pos=key_pos)
        else:
            tgt = self.forward_self_first(tgt, memory, query_pos=query_pos, key_pos=key_pos)
        return tgt


class ReDetrDecoderMultiStage(nn.Module):
    def __init__(self, dim, num_patch, num_layers, nheads, num_classes, topk, return_multi=True, cross_first=True, dim_feedforward=2048, dropout=0.1):
        super(ReDetrDecoderMultiStage, self).__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.return_multi = return_multi
        self.cls_tgt = nn.Linear(dim, num_patch)
        self.norm0 = nn.LayerNorm(dim)
        self.layer = DetrDecoderLayer(dim, nheads, topk, dim_feedforward, dropout, cross_first)
        self.layers = nn.ModuleList(copy.deepcopy(self.layer) for _ in range(num_layers))
        self.bn = BatchNorm(dim)
        self.bns = nn.ModuleList(copy.deepcopy(self.bn) for _ in range(num_layers))

        self.classifier = Classifier(dim, num_classes)
        self.classifiers = nn.ModuleList(copy.deepcopy(self.classifier) for _ in range(num_layers))

    def forward(self, tgt, memorys, query_pos=None, key_pos=None):
        bs, np, d = tgt.shape # batch_size, num_patch, dim
        tgt = self.norm0(tgt)
        if self.return_multi:
            output = []
            cls_scores = []
            output.append(tgt)
            cls = tgt.reshape(np * bs, d)
            cls = self.bn(cls)
            cls = self.classifier(cls)
            cls_scores.append(cls.reshape(bs, np, -1))

        tgt = tgt.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        key_pos = key_pos.permute(1, 0, 2)

        for i in range(self.num_layers):
            memory = memorys[i].permute(1, 0, 2)
            tgt = self.layers[i](tgt, memory, query_pos, key_pos)
            if self.return_multi:
                cls = tgt.reshape(np * bs, d)
                cls = self.bns[i](cls)
                cls = self.classifiers[i](cls)
                cls = cls.reshape(np, bs, self.num_classes)
                cls_scores.append(cls.permute(1, 0, 2))
                output.append(tgt.permute(1, 0, 2))

        if self.return_multi:
            return output, cls_scores
        else:
            cls = self.bn(tgt.reshape(np * bs, d))
            cls = self.classifier(cls)
            cls = cls.reshape(np, bs, self.num_classes)
            return tgt.permute(1, 0, 2), cls.permute(1, 0, 2)