import torch.nn.functional as F
import torch


def patchLoss(patch_feat):
    bs = patch_feat.size(0)
    num_patch = patch_feat.size(1)
    loss_exclusive = 0
    device = patch_feat.device
    for i in range(bs):
        patch_features_single_img = patch_feat[i]
        patch_features_single_img_norm = F.normalize(patch_features_single_img, p=2, dim=1)
        cosine_similarity = torch.mm(patch_features_single_img_norm, patch_features_single_img_norm.t())
        logit = F.log_softmax(cosine_similarity, dim=1)
        target_patch = torch.arange(num_patch).to(device)
        # target = torch.arange(num_patch).cuda()  # each patch belongs to a exlusive class
        loss_exclusive += F.nll_loss(logit, target_patch) / bs

        # outp_class_single = patch_cls[i]
        # # target = torch.arange(num_patch).cuda()
        # target_patch = torch.arange(num_patch).to(device)
        # outp_class_single_log = F.log_softmax(outp_class_single, dim=1)
        # zeros = torch.zeros(outp_class_single_log.size())
        # # targets = zeros.scatter_(1, target.unsqueeze(1).data.cpu(), 1).cuda()
        # targets = zeros.scatter_(1, target_patch.unsqueeze(1).data.cpu(), 1).to(device)
        # loss_exclusive += (-targets * outp_class_single_log).mean(0).sum() / bs
    return loss_exclusive

def patchLoss1(patch_feat):
    bs = patch_feat.size(0)
    num_patch = patch_feat.size(1)
    loss_exclusive = 0
    device = patch_feat.device
    for i in range(bs):
        patch_features_single_img = patch_feat[i]
        k = patch_features_single_img.size(0)
        patch_features_single_img_norm = F.normalize(patch_features_single_img, p=2, dim=1)
        cosine_similarity = torch.mm(patch_features_single_img_norm, patch_features_single_img_norm.t())
        mask = torch.eye(k).to(device)
        mask = 1 - mask
        s = cosine_similarity * mask
        loss_exclusive += torch.sum(s) / (k * (k-1))


        # outp_class_single = patch_cls[i]
        # # target = torch.arange(num_patch).cuda()
        # target_patch = torch.arange(num_patch).to(device)
        # outp_class_single_log = F.log_softmax(outp_class_single, dim=1)
        # zeros = torch.zeros(outp_class_single_log.size())
        # # targets = zeros.scatter_(1, target.unsqueeze(1).data.cpu(), 1).cuda()
        # targets = zeros.scatter_(1, target_patch.unsqueeze(1).data.cpu(), 1).to(device)
        # loss_exclusive += (-targets * outp_class_single_log).mean(0).sum() / bs
    return loss_exclusive / bs