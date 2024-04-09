import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .nformer import NFormer
from .detr_decoder import DetrDecoder, DetrDecoderMultiStage, DetrDecoderwithG
from .gcn import GCN
from .reattnDetr import ReDetrDecoder, ReDetrDecoderMultiStage
from .DeformableDetr import DeformableDetr

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

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


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

# 仅gobal
# class build_transformer(nn.Module):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory):
#         super(build_transformer, self).__init__()
#         last_stride = cfg.MODEL.LAST_STRIDE
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         model_name = cfg.MODEL.NAME
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768
#
#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
#
#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0
#
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
#                                                         camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
#                                                         drop_rate= cfg.MODEL.DROP_OUT,
#                                                         attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
#         if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
#             self.in_planes = 384
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#         self.gap = nn.AdaptiveAvgPool2d(1)
#
#         self.num_classes = num_classes
#         self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         if self.ID_LOSS_TYPE == 'arcface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Arcface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'cosface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Cosface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'amsoftmax':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'circle':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = CircleLoss(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         else:
#             self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
#             self.classifier.apply(weights_init_classifier)
#
#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.bottleneck.apply(weights_init_kaiming)
#
#     def forward(self, x, label=None, cam_label= None, view_label=None):
#         global_feat, _, _ = self.base(x, cam_label=cam_label, view_label=view_label)
#         global_feat = global_feat[-1][:, 0]
#         feat = self.bottleneck(global_feat)
#
#         if self.training:
#             if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
#                 cls_score = self.classifier(feat, label)
#             else:
#                 cls_score = self.classifier(feat)
#
#             return cls_score, global_feat  # global feature for triplet loss
#         else:
#             if self.neck_feat == 'after':
#                 # print("Test with feature after BN")
#                 return feat
#             else:
#                 # print("Test with feature before BN")
#                 return global_feat
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))
#
#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))
#
# class build_transformer_gcn(nn.Module):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory):
#         super(build_transformer_gcn, self).__init__()
#         last_stride = cfg.MODEL.LAST_STRIDE
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         model_name = cfg.MODEL.NAME
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768
#
#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
#
#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0
#
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
#                                                         camera=camera_num, view=view_num,
#                                                         stride_size=cfg.MODEL.STRIDE_SIZE,
#                                                         drop_path_rate=cfg.MODEL.DROP_PATH,
#                                                         drop_rate=cfg.MODEL.DROP_OUT,
#                                                         attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
#         self.gcn = GCN(self.in_planes, 5, 3)
#         if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
#             self.in_planes = 384
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#         self.gap = nn.AdaptiveAvgPool2d(1)
#
#         self.num_classes = num_classes
#         self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         if self.ID_LOSS_TYPE == 'arcface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
#                                                      cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Arcface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'cosface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
#                                                      cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Cosface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'amsoftmax':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
#                                                      cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'circle':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
#                                                      cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = CircleLoss(self.in_planes, self.num_classes,
#                                          s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         else:
#             self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
#             self.classifier.apply(weights_init_classifier)
#             self.classifier_gcn = nn.Linear(self.in_planes, self.num_classes, bias=False)
#             self.classifier_gcn.apply(weights_init_classifier)
#
#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.bottleneck.apply(weights_init_kaiming)
#
#         self.bottleneck_gcn = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck_gcn.bias.requires_grad_(False)
#         self.bottleneck_gcn.apply(weights_init_kaiming)
#
#     def forward(self, x, label=None, cam_label=None, view_label=None):
#         global_feat, _, _ = self.base(x, cam_label=cam_label, view_label=view_label)
#         global_feat = global_feat[-1][:, 0]
#         feat = self.bottleneck(global_feat)
#
#         gcn_feat = self.gcn(global_feat.unsqueeze(0))
#         gcn_feat = gcn_feat[0]
#         gcn_bn_feat = self.bottleneck_gcn(gcn_feat)
#
#         if self.training:
#             cls_score = self.classifier(feat)
#             cls_gcn_score = self.classifier_gcn(gcn_bn_feat)
#             return [cls_score, cls_gcn_score], [global_feat, gcn_feat]  # global feature for triplet loss
#         else:
#             if self.neck_feat == 'after':
#                 # print("Test with feature after BN")
#                 return gcn_bn_feat
#             else:
#                 # print("Test with feature before BN")
#                 return gcn_feat
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))
#
#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))
#
# class build_transformer_localmulti_gcn(nn.Module):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory):
#         super(build_transformer_localmulti_gcn, self).__init__()
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768
#         self.num_patch = cfg.MODEL.NUM_PATCH
#
#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
#
#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0
#
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
#                                                         camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
#         self.local_feat_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
#         self.query_pos_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
#         self.detr_decoder = DetrDecoderMultiStage(self.in_planes, self.num_patch, 12, 12, num_classes,
#                                                   return_multi=False, cross_first=cfg.MODEL.DECODER_CROSS_FIRST)
#         self.gcn = GCN(self.in_planes, 5, 3)
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#         self.num_classes = num_classes
#         self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         if self.ID_LOSS_TYPE == 'arcface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Arcface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'cosface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Cosface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'amsoftmax':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'circle':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = CircleLoss(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         else:
#             self.classifier = Classifier(self.in_planes, self.num_classes)
#             self.classifier_gcn = Classifier(self.in_planes, self.num_classes)
#             # self.classifier_local = nn.ModuleList([copy.deepcopy(self.classifier) for i in range(self.num_patch)])
#
#         self.bn = BatchNorm(self.in_planes)
#         # self.bn_local = nn.ModuleList([copy.deepcopy(self.bn) for i in range(self.num_patch)])
#         self.bn_gcn = BatchNorm(self.in_planes)
#
#     def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
#         bs = x.size(0)
#         device = x.device
#         features, pos = self.base(x, cam_label=cam_label, view_label=view_label)
#
#         # global
#         global_feat = features[-1][:, 0]
#
#         local_feat = self.local_feat_embed.weight
#         local_feat = local_feat.unsqueeze(0).repeat(bs, 1, 1)
#         query_pos = self.query_pos_embed.weight
#         query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1)
#
#         local_feat, cls_local = self.detr_decoder(local_feat, features, query_pos=query_pos, key_pos=pos)
#
#         feats_gcn = torch.cat([global_feat.unsqueeze(1), local_feat], dim=1)
#         bs, np, d = feats_gcn.shape
#         feats_gcn = feats_gcn.reshape(bs * np, d).unsqueeze(0)
#         feats_gcn = self.gcn(feats_gcn)[0]
#         bn_feat_gcn = self.bn_gcn(feats_gcn)
#         feats_gcn = feats_gcn.reshape(bs, np, d)
#
#         global_feat_bn = self.bn(global_feat)
#
#         if self.training:
#             cls_gcn = self.classifier_gcn(bn_feat_gcn).reshape(bs, np, -1)
#             return [self.classifier(global_feat_bn), cls_local, cls_gcn], [global_feat, local_feat, feats_gcn]
#         else:
#             # f = [local_feat[:, i, :] / self.num_patch for i in range(self.num_patch)]
#             # f.append(global_feat)
#             # return torch.cat(f, dim=1)
#             f = [feats_gcn[:, i] for i in range(self.num_patch + 1)]
#             # f = [feats_nformer[:, i+1] for i in range(self.num_patch)]
#             # f.append(feats_nformer[:, 0])
#             return torch.cat(f, dim=1)
#
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))
#
#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))
#
# # global + g_nformer
# class build_transformer_nformer(nn.Module):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory):
#         super(build_transformer_nformer, self).__init__()
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768
#         self.num_patch = cfg.MODEL.NUM_PATCH
#
#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
#
#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0
#
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
#         # self.nformer = NFormer(self.in_planes, 12, cfg.MODEL.NFORMER_LAYERS, 0.1,
#         #                        cfg.MODEL.NFORMER_TOPK, cfg.MODEL.LANDMARK, num_classes=num_classes)
#         self.nformer = DeformableDetr(self.in_planes, 4, 3)
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#
#         self.num_classes = num_classes
#         self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         if self.ID_LOSS_TYPE == 'arcface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Arcface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'cosface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Cosface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'amsoftmax':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'circle':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = CircleLoss(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         else:
#             self.classifier = Classifier(self.in_planes, self.num_classes)
#             self.classifier1 = Classifier(self.in_planes, self.num_classes)
#
#         self.bn = BatchNorm(self.in_planes)
#         self.bn1 = BatchNorm(self.in_planes)
#
#     def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
#         # bs = x.size(0)
#         # device = x.device
#         features, pos, _ = self.base(x, cam_label=cam_label, view_label=view_label)
#         # global
#         global_feat = features[-1][:, 0]
#         feat = self.bn(global_feat)
#
#         feats_nformer = global_feat.unsqueeze(1)
#         feats_nformer = self.nformer(feats_nformer)[:,0]
#         # bs, np, d = feats_nformer.shape
#         bn_feats_nformer = self.bn1(feats_nformer)
#
#         # feat_nformer, cls_score_nformer = self.nformer(global_feat.unsqueeze(0))
#
#         if self.training:
#             cls_scores = []
#             cls_scores.append(self.classifier(feat))
#             # cls_scores.append(cls_score_nformer[0])
#             cls_scores.append(self.classifier1(bn_feats_nformer))
#
#             return cls_scores, [global_feat, feats_nformer]
#         else:
#             # f = [global_feat, feat_nformer]
#             # return torch.cat(f, dim=1)
#             return feats_nformer
#
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))
#
#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))
#
# class build_transformer_nformermulti(nn.Module):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory):
#         super(build_transformer_nformermulti, self).__init__()
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768
#         self.num_patch = cfg.MODEL.NUM_PATCH
#
#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
#
#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0
#
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
#         self.nformer = NFormer(self.in_planes, 12, cfg.MODEL.NFORMER_LAYERS, 0.1,
#                                cfg.MODEL.NFORMER_TOPK, cfg.MODEL.LANDMARK, num_classes=num_classes)
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#
#         self.num_classes = num_classes
#         self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         if self.ID_LOSS_TYPE == 'arcface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Arcface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'cosface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Cosface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'amsoftmax':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'circle':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = CircleLoss(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         else:
#             self.classifier = Classifier(self.in_planes, self.num_classes)
#
#         self.bn = BatchNorm(self.in_planes)
#
#     def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
#         # bs = x.size(0)
#         # device = x.device
#         features, pos = self.base(x, cam_label=cam_label, view_label=view_label)
#         # global
#         global_feat = features[-1][:, 0]
#         feat = self.bn(global_feat)
#
#         feat_nformer, cls_score_nformer = self.nformer(global_feat.unsqueeze(0))
#
#         if self.training:
#             cls_scores = []
#             cls_scores.append(self.classifier(feat))
#             cls_scores.append(cls_score_nformer[0])
#
#             return cls_scores, [global_feat, feat_nformer[0]]
#         else:
#             # f = [global_feat, feat_nformer]
#             # return torch.cat(f, dim=1)
#             return feat_nformer[0]
#
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))
#
#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))
#
# # global + local
# class build_transformer_local(nn.Module):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory):
#         super(build_transformer_local, self).__init__()
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768
#         self.num_patch = cfg.MODEL.NUM_PATCH
#         self.local_feat_init_0 = cfg.MODEL.LOCAL_FEAT_INIT_0
#
#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
#
#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0
#
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
#                                                         camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
#         self.local_feat_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
#         self.query_pos_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
#         self.detr_decoder = DetrDecoder(self.in_planes, self.num_patch, cfg.MODEL.DECODER_LAYERS, 12, num_classes,
#                                             return_multi=False, cross_first=cfg.MODEL.DECODER_CROSS_FIRST)
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#         self.num_classes = num_classes
#         self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         if self.ID_LOSS_TYPE == 'arcface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Arcface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'cosface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Cosface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'amsoftmax':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'circle':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = CircleLoss(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         else:
#             self.classifier = Classifier(self.in_planes, self.num_classes)
#
#         self.bn = BatchNorm(self.in_planes)
#
#     def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
#         bs = x.size(0)
#         device = x.device
#         features, pos = self.base(x, cam_label=cam_label, view_label=view_label)
#
#         # global
#         global_feat = features[-1][:, 0]
#
#         # local
#         if self.local_feat_init_0:
#             local_feat = torch.zeros((bs, self.num_patch, self.in_planes)).to(device)
#         else:
#             local_feat = self.local_feat_embed.weight
#             local_feat = local_feat.unsqueeze(0).repeat(bs, 1, 1)
#
#         query_pos = self.query_pos_embed.weight
#         query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1)
#
#         local_feat, cls_pred = self.detr_decoder(local_feat, features[-1], query_pos=query_pos, key_pos=pos)
#
#         global_feat_bn = self.bn(global_feat)
#
#         if self.training:
#             return [self.classifier(global_feat_bn), cls_pred], [global_feat, local_feat]
#         else:
#             f = [local_feat[:, i, :] / self.num_patch for i in range(self.num_patch)]
#             f.append(global_feat)
#             return torch.cat(f, dim=1)
#
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))
#
#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))

# class build_transformer_local2(nn.Module):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory):
#         super(build_transformer_local2, self).__init__()
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768
#         self.num_patch = cfg.MODEL.NUM_PATCH
#         self.local_feat_init_0 = cfg.MODEL.LOCAL_FEAT_INIT_0
#
#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
#
#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0
#
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
#                                                         camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
#         self.local_feat_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
#         self.query_pos_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
#         self.detr_decoder = DetrDecoderwithG(self.in_planes, self.num_patch, cfg.MODEL.DECODER_LAYERS, 12, num_classes,
#                                             return_multi=False, cross_first=cfg.MODEL.DECODER_CROSS_FIRST)
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#         self.num_classes = num_classes
#         self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         if self.ID_LOSS_TYPE == 'arcface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Arcface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'cosface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Cosface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'amsoftmax':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'circle':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = CircleLoss(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         else:
#             self.classifier = Classifier(self.in_planes, self.num_classes)
#
#         self.bn = BatchNorm(self.in_planes)
#
#     def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
#         bs = x.size(0)
#         device = x.device
#         features, pos = self.base(x, cam_label=cam_label, view_label=view_label)
#
#         # global
#         global_feat = features[-1][:, 0]
#
#         # local
#         if self.local_feat_init_0:
#             local_feat = torch.zeros((bs, self.num_patch, self.in_planes)).to(device)
#         else:
#             local_feat = self.local_feat_embed.weight
#             local_feat = local_feat.unsqueeze(0).repeat(bs, 1, 1)
#
#         query_pos = self.query_pos_embed.weight
#         query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1)
#
#         local_feat, cls_pred = self.detr_decoder(local_feat, features[-1], query_pos=query_pos, key_pos=pos)
#
#         global_feat_bn = self.bn(global_feat)
#
#         if self.training:
#             return [self.classifier(global_feat_bn), cls_pred], [global_feat, local_feat]
#         else:
#             f = [local_feat[:, i, :] / self.num_patch for i in range(self.num_patch)]
#             f.append(global_feat)
#             return torch.cat(f, dim=1)
#
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))
#
#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))
#
# class build_transformer_local3(nn.Module):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory):
#         super(build_transformer_local3, self).__init__()
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768
#         self.num_patch = cfg.MODEL.NUM_PATCH
#
#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
#
#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0
#
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
#                                                         camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
#         # self.local_feat_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
#         self.query_pos_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
#         # self.detr_decoder = DetrDecoderMultiStage(self.in_planes, self.num_patch, 12, 12, num_classes,
#         #                                           return_multi=False, cross_first=cfg.MODEL.DECODER_CROSS_FIRST)
#         self.detr_decoder = ReDetrDecoderMultiStage(self.in_planes, self.num_patch, 12, 12, num_classes,
#                                                   50, return_multi=False, cross_first=cfg.MODEL.DECODER_CROSS_FIRST)
#         # self.detr_decoder = ReDetrDecoder(self.in_planes, self.num_patch, 6, 12, num_classes,
#         #                                             50, return_multi=False, cross_first=cfg.MODEL.DECODER_CROSS_FIRST)
#         self.gap1 = nn.AdaptiveAvgPool2d(1)
#         self.gap2 = nn.AdaptiveAvgPool2d(1)
#         self.gap3 = nn.AdaptiveAvgPool2d(1)
#         self.gap4 = nn.AdaptiveAvgPool2d(1)
#         # self.gap5 = nn.AdaptiveAvgPool2d(1)
#
#         # self.gap1 = nn.AdaptiveMaxPool2d(1)
#         # self.gap2 = nn.AdaptiveMaxPool2d(1)
#         # self.gap3 = nn.AdaptiveMaxPool2d(1)
#         # self.gap4 = nn.AdaptiveMaxPool2d(1)
#         # self.gap5 = nn.AdaptiveMaxPool2d(1)
#         # self.gap6 = nn.AdaptiveMaxPool2d(1)
#         # self.gap7 = nn.AdaptiveMaxPool2d(1)
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#         self.num_classes = num_classes
#         self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         if self.ID_LOSS_TYPE == 'arcface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Arcface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'cosface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Cosface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'amsoftmax':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'circle':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = CircleLoss(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         else:
#             self.classifier = Classifier(self.in_planes, self.num_classes)
#             # self.classifier_local = nn.ModuleList([copy.deepcopy(self.classifier) for i in range(self.num_patch)])
#
#         self.bn = BatchNorm(self.in_planes)
#         # self.bn_local = nn.ModuleList([copy.deepcopy(self.bn) for i in range(self.num_patch)])
#
#
#     def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
#         bs = x.size(0)
#         device = x.device
#         features, pos, f_patch = self.base(x, cam_label=cam_label, view_label=view_label)
#
#         # global
#         global_feat = features[-1][:, 0]
#
#         l1 = self.gap1(f_patch[:,:,0:6]).view(bs, -1).unsqueeze(1)
#         l2 = self.gap2(f_patch[:,:,6:11]).view(bs, -1).unsqueeze(1)
#         l3 = self.gap3(f_patch[:,:,11:16]).view(bs, -1).unsqueeze(1)
#         l4 = self.gap4(f_patch[:,:,16:]).view(bs, -1).unsqueeze(1)
#
#         # l1 = self.gap1(f_patch[:,:,0:6]).view(bs, -1).unsqueeze(1)
#         # l2 = self.gap2(f_patch[:,:,6:12]).view(bs, -1).unsqueeze(1)
#         # l3 = self.gap3(f_patch[:,:,12:17]).view(bs, -1).unsqueeze(1)
#         # l4 = self.gap4(f_patch[:,:,17:]).view(bs, -1).unsqueeze(1)
#
#         # l1 = self.gap1(f_patch[:, :, 0:3]).view(bs, -1).unsqueeze(1)
#         # l2 = self.gap2(f_patch[:, :, 3:6]).view(bs, -1).unsqueeze(1)
#         # l3 = self.gap3(f_patch[:, :, 6:9]).view(bs, -1).unsqueeze(1)
#         # l4 = self.gap4(f_patch[:, :, 9:12]).view(bs, -1).unsqueeze(1)
#         # l5 = self.gap5(f_patch[:, :, 12:15]).view(bs, -1).unsqueeze(1)
#         # l6 = self.gap6(f_patch[:, :, 15:18]).view(bs, -1).unsqueeze(1)
#         # l7 = self.gap7(f_patch[:, :, 18:]).view(bs, -1).unsqueeze(1)
#
#         local_feat = torch.cat([l1, l2, l3, l4], dim=1)
#
#         # local_feat = self.local_feat_embed.weight
#         # local_feat = local_feat.unsqueeze(0).repeat(bs, 1, 1)
#         # query_pos = self.query_pos_embed.weight
#         # query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1)
#
#         local_feat, cls_local = self.detr_decoder(local_feat, features, query_pos=None, key_pos=pos)
#         # local_feat, cls_local = self.detr_decoder(local_feat, features[-1], query_pos=query_pos, key_pos=pos)
#
#         global_feat_bn = self.bn(global_feat)
#
#         if self.training:
#             return [self.classifier(global_feat_bn), cls_local], [global_feat, local_feat]
#         else:
#             f = [local_feat[:, i, :] / self.num_patch for i in range(self.num_patch)]
#             f.append(global_feat)
#             return torch.cat(f, dim=1)
#
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))
#
#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))
# class build_transformer_local3_nformer(nn.Module):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory):
#         super(build_transformer_local3_nformer, self).__init__()
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768
#         self.num_patch = cfg.MODEL.NUM_PATCH
#
#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
#
#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0
#
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
#                                                         camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
#         self.local_feat_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
#         self.query_pos_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
#         # self.detr_decoder = DetrDecoderMultiStage(self.in_planes, self.num_patch, 12, 12, num_classes,
#         #                                           return_multi=False, cross_first=cfg.MODEL.DECODER_CROSS_FIRST)
#         self.detr_decoder = ReDetrDecoderMultiStage(self.in_planes, self.num_patch, 12, 12, num_classes,
#                                                   50, return_multi=False, cross_first=cfg.MODEL.DECODER_CROSS_FIRST)
#         # self.detr_decoder = ReDetrDecoder(self.in_planes, self.num_patch, 6, 12, num_classes,
#         #                                             50, return_multi=False, cross_first=cfg.MODEL.DECODER_CROSS_FIRST)
#         self.nformer = NFormer(self.in_planes, 12, cfg.MODEL.NFORMER_LAYERS, 0.1,
#                                cfg.MODEL.NFORMER_TOPK, cfg.MODEL.LANDMARK, num_classes=num_classes)
#         # self.gap1 = nn.AdaptiveAvgPool2d(1)
#         # self.gap2 = nn.AdaptiveAvgPool2d(1)
#         # self.gap3 = nn.AdaptiveAvgPool2d(1)
#         # self.gap4 = nn.AdaptiveAvgPool2d(1)
#         # self.gap5 = nn.AdaptiveAvgPool2d(1)
#
#         # self.gap1 = nn.AdaptiveMaxPool2d(1)
#         # self.gap2 = nn.AdaptiveMaxPool2d(1)
#         # self.gap3 = nn.AdaptiveMaxPool2d(1)
#         # self.gap4 = nn.AdaptiveMaxPool2d(1)
#         # self.gap5 = nn.AdaptiveMaxPool2d(1)
#         # self.gap6 = nn.AdaptiveMaxPool2d(1)
#         # self.gap7 = nn.AdaptiveMaxPool2d(1)
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#         self.num_classes = num_classes
#         self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         if self.ID_LOSS_TYPE == 'arcface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Arcface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'cosface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Cosface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'amsoftmax':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'circle':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = CircleLoss(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         else:
#             self.classifier = Classifier(self.in_planes, self.num_classes)
#             # self.classifier_local = nn.ModuleList([copy.deepcopy(self.classifier) for i in range(self.num_patch)])
#
#         self.bn = BatchNorm(self.in_planes)
#         # self.bn_local = nn.ModuleList([copy.deepcopy(self.bn) for i in range(self.num_patch)])
#
#
#     def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
#         bs = x.size(0)
#         device = x.device
#         features, pos, f_patch = self.base(x, cam_label=cam_label, view_label=view_label)
#
#         # global
#         global_feat = features[-1][:, 0]
#
#         # l1 = self.gap1(f_patch[:,:,0:6]).view(bs, -1).unsqueeze(1)
#         # l2 = self.gap2(f_patch[:,:,6:11]).view(bs, -1).unsqueeze(1)
#         # l3 = self.gap3(f_patch[:,:,11:16]).view(bs, -1).unsqueeze(1)
#         # l4 = self.gap4(f_patch[:,:,16:]).view(bs, -1).unsqueeze(1)
#
#         # l1 = self.gap1(f_patch[:,:,0:6]).view(bs, -1).unsqueeze(1)
#         # l2 = self.gap2(f_patch[:,:,6:12]).view(bs, -1).unsqueeze(1)
#         # l3 = self.gap3(f_patch[:,:,12:17]).view(bs, -1).unsqueeze(1)
#         # l4 = self.gap4(f_patch[:,:,17:]).view(bs, -1).unsqueeze(1)
#
#         # l1 = self.gap1(f_patch[:, :, 0:3]).view(bs, -1).unsqueeze(1)
#         # l2 = self.gap2(f_patch[:, :, 3:6]).view(bs, -1).unsqueeze(1)
#         # l3 = self.gap3(f_patch[:, :, 6:9]).view(bs, -1).unsqueeze(1)
#         # l4 = self.gap4(f_patch[:, :, 9:12]).view(bs, -1).unsqueeze(1)
#         # l5 = self.gap5(f_patch[:, :, 12:15]).view(bs, -1).unsqueeze(1)
#         # l6 = self.gap6(f_patch[:, :, 15:18]).view(bs, -1).unsqueeze(1)
#         # l7 = self.gap7(f_patch[:, :, 18:]).view(bs, -1).unsqueeze(1)
#
#         # local_feat = torch.cat([l1, l2, l3, l4], dim=1)
#
#         local_feat = self.local_feat_embed.weight
#         local_feat = local_feat.unsqueeze(0).repeat(bs, 1, 1)
#         query_pos = self.query_pos_embed.weight
#         query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1)
#
#         local_feat, cls_local = self.detr_decoder(local_feat, features, query_pos=query_pos, key_pos=pos)
#         # local_feat, cls_local = self.detr_decoder(local_feat, features[-1], query_pos=query_pos, key_pos=pos)
#
#         feats_nformer = torch.cat([global_feat.unsqueeze(1), local_feat], dim=1)
#         bs, np, d = feats_nformer.shape
#         feats_nformer = feats_nformer.reshape(bs * np, d).unsqueeze(0)
#         feats_nformer, cls_nformer = self.nformer(feats_nformer)
#         feats_nformer = feats_nformer.view(bs, np, d)
#         cls_nformer = cls_nformer.view(bs, np, -1)
#
#         global_feat_bn = self.bn(global_feat)
#
#         if self.training:
#             return [self.classifier(global_feat_bn), cls_local, cls_nformer], [global_feat, local_feat, feats_nformer]
#         else:
#             # f = [local_feat[:, i, :] / self.num_patch for i in range(self.num_patch)]
#             # f.append(global_feat)
#             # return torch.cat(f, dim=1)
#             f = [feats_nformer[:, i] for i in range(self.num_patch + 1)]
#             # f = [feats_nformer[:, i+1] for i in range(self.num_patch)]
#             # f.append(feats_nformer[:, 0])
#             return torch.cat(f, dim=1)
#
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))
#
#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))
class build_transformer_local3_deformableDetr(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer_local3_deformableDetr, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.num_patch = cfg.MODEL.NUM_PATCH

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
        self.local_feat_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
        self.query_pos_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
        # self.detr_decoder = DetrDecoderMultiStage(self.in_planes, self.num_patch, 12, 12, num_classes,
        #                                           return_multi=False, cross_first=cfg.MODEL.DECODER_CROSS_FIRST)
        self.detr_decoder = ReDetrDecoderMultiStage(self.in_planes, self.num_patch, 12, 12, num_classes,
                                                  50, return_multi=False, cross_first=cfg.MODEL.DECODER_CROSS_FIRST)
        # self.detr_decoder = ReDetrDecoder(self.in_planes, self.num_patch, 6, 12, num_classes,
        #                                             50, return_multi=False, cross_first=cfg.MODEL.DECODER_CROSS_FIRST)
        self.nformer = DeformableDetr(self.in_planes, 4, 3)
        # self.gap1 = nn.AdaptiveAvgPool2d(1)
        # self.gap2 = nn.AdaptiveAvgPool2d(1)
        # self.gap3 = nn.AdaptiveAvgPool2d(1)
        # self.gap4 = nn.AdaptiveAvgPool2d(1)
        # self.gap5 = nn.AdaptiveAvgPool2d(1)

        # self.gap1 = nn.AdaptiveMaxPool2d(1)
        # self.gap2 = nn.AdaptiveMaxPool2d(1)
        # self.gap3 = nn.AdaptiveMaxPool2d(1)
        # self.gap4 = nn.AdaptiveMaxPool2d(1)
        # self.gap5 = nn.AdaptiveMaxPool2d(1)
        # self.gap6 = nn.AdaptiveMaxPool2d(1)
        # self.gap7 = nn.AdaptiveMaxPool2d(1)
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = Classifier(self.in_planes, self.num_classes)
            self.classifier1 = Classifier(self.in_planes, self.num_classes)
            # self.classifier_local = nn.ModuleList([copy.deepcopy(self.classifier) for i in range(self.num_patch)])

        self.bn = BatchNorm(self.in_planes)
        # self.bn_local = nn.ModuleList([copy.deepcopy(self.bn) for i in range(self.num_patch)])

        self.bn1 = BatchNorm(self.in_planes)

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        bs = x.size(0)
        device = x.device
        features, pos, f_patch = self.base(x, cam_label=cam_label, view_label=view_label)

        # global
        global_feat = features[-1][:, 0]

        # l1 = self.gap1(f_patch[:,:,0:6]).view(bs, -1).unsqueeze(1)
        # l2 = self.gap2(f_patch[:,:,6:11]).view(bs, -1).unsqueeze(1)
        # l3 = self.gap3(f_patch[:,:,11:16]).view(bs, -1).unsqueeze(1)
        # l4 = self.gap4(f_patch[:,:,16:]).view(bs, -1).unsqueeze(1)

        # l1 = self.gap1(f_patch[:,:,0:6]).view(bs, -1).unsqueeze(1)
        # l2 = self.gap2(f_patch[:,:,6:12]).view(bs, -1).unsqueeze(1)
        # l3 = self.gap3(f_patch[:,:,12:17]).view(bs, -1).unsqueeze(1)
        # l4 = self.gap4(f_patch[:,:,17:]).view(bs, -1).unsqueeze(1)

        # l1 = self.gap1(f_patch[:, :, 0:3]).view(bs, -1).unsqueeze(1)
        # l2 = self.gap2(f_patch[:, :, 3:6]).view(bs, -1).unsqueeze(1)
        # l3 = self.gap3(f_patch[:, :, 6:9]).view(bs, -1).unsqueeze(1)
        # l4 = self.gap4(f_patch[:, :, 9:12]).view(bs, -1).unsqueeze(1)
        # l5 = self.gap5(f_patch[:, :, 12:15]).view(bs, -1).unsqueeze(1)
        # l6 = self.gap6(f_patch[:, :, 15:18]).view(bs, -1).unsqueeze(1)
        # l7 = self.gap7(f_patch[:, :, 18:]).view(bs, -1).unsqueeze(1)

        # local_feat = torch.cat([l1, l2, l3, l4], dim=1)

        local_feat = self.local_feat_embed.weight
        local_feat = local_feat.unsqueeze(0).repeat(bs, 1, 1)
        query_pos = self.query_pos_embed.weight
        query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1)
        #
        local_feat, cls_local = self.detr_decoder(local_feat, features, query_pos=query_pos, key_pos=pos)
        # local_feat, cls_local = self.detr_decoder(local_feat, features[-1], query_pos=query_pos, key_pos=pos)

        feats_nformer = torch.cat([global_feat.unsqueeze(1), local_feat], dim=1)
        feats_nformer = self.nformer(feats_nformer)
        bs, np, d = feats_nformer.shape
        bn_feats_nformer = self.bn1(feats_nformer.reshape(bs * np, d))


        global_feat_bn = self.bn(global_feat)

        if self.training:
            cls_nformer = self.classifier1(bn_feats_nformer)
            cls_nformer = cls_nformer.view(bs, np, -1)
            return [self.classifier(global_feat_bn), cls_local, cls_nformer], [global_feat, local_feat, feats_nformer]
        else:
            # f = [local_feat[:, i, :] / self.num_patch for i in range(self.num_patch)]
            # f.append(global_feat)
            # return torch.cat(f, dim=1)
            f = [feats_nformer[:, i] for i in range(self.num_patch + 1)]
            # f = [feats_nformer[:, i+1] for i in range(self.num_patch)]
            # f.append(feats_nformer[:, 0])
            return torch.cat(f, dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

# class build_transformer_local_multilevel(nn.Module):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory):
#         super(build_transformer_local_multilevel, self).__init__()
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768
#         self.num_patch = cfg.MODEL.NUM_PATCH
#
#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
#
#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0
#
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
#                                                         camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
#         self.local_feat_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
#         self.query_pos_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
#         self.detr_decoder = DetrDecoderMultiStage(self.in_planes, self.num_patch, 12, 12, num_classes,
#                                                   return_multi=False, cross_first=cfg.MODEL.DECODER_CROSS_FIRST)
#         self.gap1 = nn.AdaptiveAvgPool2d(1)
#         self.gap2 = nn.AdaptiveAvgPool2d(1)
#         self.gap3 = nn.AdaptiveAvgPool2d(1)
#         self.gap4 = nn.AdaptiveAvgPool2d(1)
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#         self.num_classes = num_classes
#         self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         if self.ID_LOSS_TYPE == 'arcface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Arcface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'cosface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Cosface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'amsoftmax':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'circle':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = CircleLoss(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         else:
#             self.classifier = Classifier(self.in_planes, self.num_classes)
#             # self.classifier_local = nn.ModuleList([copy.deepcopy(self.classifier) for i in range(self.num_patch)])
#
#         self.bn = BatchNorm(self.in_planes)
#         # self.bn_local = nn.ModuleList([copy.deepcopy(self.bn) for i in range(self.num_patch)])
#
#
#     def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
#         bs = x.size(0)
#         device = x.device
#         features, pos, f_patch = self.base(x, cam_label=cam_label, view_label=view_label)
#
#         # global
#         global_feat = features[-1][:, 0]
#
#         l1 = self.gap1(f_patch[:,:,0:6]).view(bs, -1).unsqueeze(1)
#         l2 = self.gap2(f_patch[:,:,6:11]).view(bs, -1).unsqueeze(1)
#         l3 = self.gap3(f_patch[:,:,11:16]).view(bs, -1).unsqueeze(1)
#         l4 = self.gap4(f_patch[:,:,16:]).view(bs, -1).unsqueeze(1)
#
#         local_feat = torch.cat([l1, l2, l3, l4], dim=1)
#
#         # local_feat = self.local_feat_embed.weight
#         # local_feat = local_feat.unsqueeze(0).repeat(bs, 1, 1)
#         query_pos = self.query_pos_embed.weight
#         query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1)
#
#         local_feat, cls_local = self.detr_decoder(local_feat, features, query_pos=query_pos, key_pos=pos)
#
#
#         global_feat_bn = self.bn(global_feat)
#
#         if self.training:
#             return [self.classifier(global_feat_bn), cls_local], [global_feat, local_feat]
#         else:
#             f = [local_feat[:, i, :] / self.num_patch for i in range(self.num_patch)]
#             f.append(global_feat)
#             return torch.cat(f, dim=1)
#
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))
#
#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))
# class build_transformer_local_nformer(nn.Module):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory):
#         super(build_transformer_local_nformer, self).__init__()
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768
#         self.num_patch = cfg.MODEL.NUM_PATCH
#
#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
#
#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0
#
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
#                                                         camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
#         self.local_feat_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
#         self.query_pos_embed = nn.Embedding(cfg.MODEL.NUM_PATCH, self.in_planes)
#         self.detr_decoder = DetrDecoderMultiStage(self.in_planes, self.num_patch, 12, 12, num_classes, return_multi=False)
#         self.nformer = NFormer(self.in_planes, 12, cfg.MODEL.NFORMER_LAYERS, 0.1,
#                                 cfg.MODEL.NFORMER_TOPK, cfg.MODEL.LANDMARK, num_classes=num_classes)
#         # self.cross_attn = CrossAttention(self.in_planes, 12, 0.1, 2048, cfg.MODEL.CROSS_LAYERS, cfg.MODEL.CROSS_NUM_PERSONS)
#
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#         self.num_classes = num_classes
#         self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         if self.ID_LOSS_TYPE == 'arcface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Arcface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'cosface':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = Cosface(self.in_planes, self.num_classes,
#                                       s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'amsoftmax':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         elif self.ID_LOSS_TYPE == 'circle':
#             print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
#             self.classifier = CircleLoss(self.in_planes, self.num_classes,
#                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         else:
#             self.classifier = Classifier(self.in_planes, self.num_classes)
#             # self.classifier_local = nn.ModuleList([copy.deepcopy(self.classifier) for i in range(self.num_patch)])
#
#         self.bn = BatchNorm(self.in_planes)
#         # self.bn_local = nn.ModuleList([copy.deepcopy(self.bn) for i in range(self.num_patch)])
#
#
#     def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
#         bs = x.size(0)
#         device = x.device
#         features, pos = self.base(x, cam_label=cam_label, view_label=view_label)
#
#         # global
#         global_feat = features[-1][:, 0]
#
#         local_feat = self.local_feat_embed.weight
#         local_feat = local_feat.unsqueeze(0).repeat(bs, 1, 1)
#         query_pos = self.query_pos_embed.weight
#         query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1)
#
#         local_feat, cls_local = self.detr_decoder(local_feat, features, query_pos=query_pos, key_pos=pos)
#
#         feats_nformer = torch.cat([global_feat.unsqueeze(1), local_feat], dim=1)
#         bs, np, d = feats_nformer.shape
#         feats_nformer = feats_nformer.reshape(bs * np, d).unsqueeze(0)
#         feats_nformer, cls_nformer = self.nformer(feats_nformer)
#         feats_nformer = feats_nformer.view(bs, np, d)
#         cls_nformer = cls_nformer.view(bs, np, -1)
#
#         global_feat_bn = self.bn(global_feat)
#
#         if self.training:
#             return [self.classifier(global_feat_bn), cls_local, cls_nformer], [global_feat, local_feat, feats_nformer]
#         else:
#             # f = [local_feat[:, i, :] / self.num_patch for i in range(self.num_patch)]
#             # f.append(global_feat)
#             # return torch.cat(f, dim=1)
#             f = [feats_nformer[:, i] for i in range(self.num_patch+1)]
#             # f = [feats_nformer[:, i+1] for i in range(self.num_patch)]
#             # f.append(feats_nformer[:, 0])
#             return torch.cat(f, dim=1)
#             # return feats_nformer[:, 0]
#
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))
#
#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))

__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    # if cfg.MODEL.IF_LOCAL and cfg.MODEL.IF_NFORMER:
    #     model = build_transformer_local3_deformableDetr(num_class, camera_num, view_num, cfg, __factory_T_type)
    #     print("===========builing transformer with local & nformer ===============")
    # elif cfg.MODEL.IF_LOCAL and cfg.MODEL.IF_GCN:
    #     model = build_transformer_localmulti_gcn(num_class, camera_num, view_num, cfg, __factory_T_type)
    #     print("===========builing transformer with local & gcn ===============")
    # elif cfg.MODEL.IF_LOCAL:
    #     if cfg.MODEL.IF_MULTILEVEL_MEMORY:
    #         model = build_transformer_local3(num_class, camera_num, view_num, cfg, __factory_T_type)
    #         print("===========builing transformer with local multi_level memory ===============")
    #     else:
    #         model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type)
    #         print("===========builing transformer with local ===============")
    # elif cfg.MODEL.IF_NFORMER:
    #     model = build_transformer_nformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    #     print("===========builing transformer with  nformer ===============")
    # elif cfg.MODEL.IF_GCN:
    #     model = build_transformer_gcn(num_class, camera_num, view_num, cfg, __factory_T_type)
    #     print("===========builing transformer with gcn ===============")
    # else:
    #     model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    #     print("===========builing transformer ===============")

    model = build_transformer_local3_deformableDetr(num_class, camera_num, view_num, cfg, __factory_T_type)

    return model
