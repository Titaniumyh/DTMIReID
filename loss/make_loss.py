# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .patch_loss import patchLoss, patchLoss1


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=False)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                # if cfg.MODEL.IF_LOCAL and cfg.MODEL.IF_NFORMER:
                PATCH_LOSS = patchLoss(feat[1])
                # PATCH_LOSS = 0
                ID_LOSS = [F.cross_entropy(score[1][:, i], target) for i in range(score[1].size(1))]
                ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                ID_LOSS1 = [F.cross_entropy(score[2][:, i], target) for i in range(score[2].size(1))]
                ID_LOSS1 = sum(ID_LOSS1) / len(ID_LOSS1)
                ID_LOSS = 0.5 * ID_LOSS + 0.5 * ID_LOSS1

                TRI_LOSS = [triplet(feat[1][:, i, :], target)[0] for i in range(feat[1].size(1))]
                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                TRI_LOSS1 = [triplet(feat[2][:, i, :], target)[0] for i in range(feat[2].size(1))]
                TRI_LOSS1 = sum(TRI_LOSS1) / len(TRI_LOSS1)
                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * TRI_LOSS1
                # elif cfg.MODEL.IF_LOCAL:
                #     PATCH_LOSS = patchLoss(feat[1])
                #
                #     ID_LOSS = [F.cross_entropy(score[1][:, i], target) for i in range(score[1].size(1))]
                #     ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                #     ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                #
                #     TRI_LOSS = [triplet(feat[1][:, i, :], target)[0] for i in range(feat[1].size(1))]
                #     TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                #     TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                # elif cfg.MODEL.IF_NFORMER:
                #     PATCH_LOSS = 0
                #     ID_LOSS = 0.5 * F.cross_entropy(score[0], target) + 0.5 * F.cross_entropy(score[1], target)
                #     TRI_LOSS = 0.5 * triplet(feat[0], target)[0] + 0.5 * triplet(feat[1], target)[0]
                # elif cfg.MODEL.IF_GCN:
                #     PATCH_LOSS = 0
                #     ID_LOSS = 0.5 * F.cross_entropy(score[0], target) + 0.5 * F.cross_entropy(score[1], target)
                #     TRI_LOSS = 0.5 * triplet(feat[0], target)[0] + 0.5 * triplet(feat[1], target)[0]
                # else:
                #     PATCH_LOSS = 0
                #     ID_LOSS = F.cross_entropy(score, target)
                #     TRI_LOSS = triplet(feat, target)[0]

                return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                       cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + PATCH_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


