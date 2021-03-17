from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def iou(logit, target, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        logit, target = (logit,), (target,)
    ious = []
    for pred, label in zip(logit, target):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)

def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

# --------------------------- MULTICLASS LOSSES ---------------------------

def lovasz_softmax_flat(logit, target, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      logit: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      target: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if logit.numel() == 0:
        # only void pixels, the gradients should be 0
        return logit * 0.
    C = logit.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (target == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = logit[:, 0]
        else:
            class_pred = logit[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_logit(logit, target, ignore=None):
    """
    Flattens predictions in the batch
    """
    n, c, h, w = logit.size()
    logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, c)  # b * h * w, c = P, C
    target = target.view(-1)
    if ignore is None:
        return logit, target
    valid = (target != ignore)
    vlogit = logit[valid.nonzero().squeeze()]
    vtarget = target[valid]
    return vlogit, vtarget


class SegmentationLosses(object):
    def __init__(self, weight=None, reduction='mean', batch_average=True, ignore_index=255, cuda=False):    # size_average=True
        self.ignore_index = ignore_index
        self.weight = weight
        # self.size_average = size_average
        self.reduction = reduction
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal' or 'lovasz' or 'ce+focal+lovasz']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'lovasz':
            return self.LovaszSoftmaxLoss
        elif mode == 'dice':
            return self.DiceLoss
        elif mode == 'ce+focal':
            return self.Ce_Focal_Loss
        #elif mode == 'ce+focal+lovasz+dice':
        #    return self.CrossEntropyLoss + self.FocalLoss + self.LovaszSoftmaxLoss + self.DiceLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction)

        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2.0, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def Ce_Focal_Loss(self, logit, target, gamma=2.0, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction)

        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss_focal = -((1 - pt) ** gamma) * logpt

        loss_ce = criterion(logit, target.long())

        loss = loss_focal + loss_ce

        if self.batch_average:
            loss /= n

        return loss
    
    def DiceLoss(self, logit, target, gamma=2.0, alpha=0.5):    # alpha = 1.0, gamma = 1.0
        n, c, h, w = logit.size()
        target = target.long()
        prob = torch.softmax(logit, dim=1)
        prob = torch.gather(prob, dim=1, index=target.unsqueeze(1))

        prob_with_factor = ((1 - prob) ** alpha) * prob
        criterion = (1 - (2 * prob_with_factor + gamma) / (prob_with_factor + 1 + gamma)).mean()
        
        if self.cuda:
            criterion = criterion.cuda()
            
        # loss = criterion(logit, target.long())
        loss = criterion

        if self.batch_average:
            loss /= n

        return loss


    def LovaszSoftmaxLoss(self, logit, target, classes='present', per_image=False, ignore=255):
    # def LovaszSoftmaxLoss(self, logit, target, classes='present', per_image=False, ignore=self.ignore_index):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                  Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """
        n, c, h, w = logit.size()
        target = target.long()
        if per_image:
            criterion = mean(lovasz_softmax_flat(*flatten_logit(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                              for prob, lab in zip(logit, target))
        else:
            criterion = lovasz_softmax_flat(*flatten_logit(logit, target, ignore), classes=classes)
            
        if self.cuda:
            criterion = criterion.cuda()
            
        # loss = criterion(logit, target.long())
        loss = criterion
        
        if self.batch_average:
            loss /= n
        
        return loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    ce = loss.CrossEntropyLoss(a, b).item()
    dice = loss.DiceLoss(a, b, gamma=2, alpha=0.5).item()
    lovasz = loss.LovaszSoftmaxLoss(a, b).item()
    focal = loss.FocalLoss(a, b, gamma=2, alpha=0.5).item()
    print(ce)
    print(lovasz)
    print(focal)
    print(dice)
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(ce+lovasz+focal+dice)




