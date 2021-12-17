#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..utils import logging
logger = logging.get_logger("nearest_neighbors")


class SigmoidLoss(nn.Module):
    def __init__(self, cfg=None):
        super(SigmoidLoss, self).__init__()

    def is_single(self):
        return True

    def multi_hot(self, labels: torch.Tensor, nb_classes: int) -> torch.Tensor:
        # logger.info(labels)
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        # logger.info(labels)
        target = torch.zeros(
            labels.size(0), nb_classes, device=labels.device
        ).scatter_(1, labels, 1.)
        # (batch_size, num_classes)
        return target

    def loss(
        self, logits, targets, per_cls_weights,
        multihot_targets: Optional[bool] = False
    ):
        # targets: 1d-tensor of integer
        # Only support single label at this moment
        # if len(targets.shape) != 2:
        num_classes = logits.shape[1]
        targets = self.multi_hot(targets, num_classes)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        # logger.info(f"loss shape: {loss.shape}")
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        ).unsqueeze(0)
        # logger.info(f"weight shape: {weight.shape}")
        loss = torch.mul(loss.to(torch.float32), weight.to(torch.float32))
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, targets, per_cls_weights, multihot_targets=False
    ):
        loss = self.loss(
            pred_logits, targets,  per_cls_weights, multihot_targets)
        return loss


class SoftmaxLoss(SigmoidLoss):
    def __init__(self, cfg=None):
        super(SoftmaxLoss, self).__init__()

    def loss(self, logits, targets, per_cls_weights, kwargs):
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        )
        loss = F.cross_entropy(logits, targets, weight, reduction="none")

        return torch.sum(loss) / targets.shape[0]


class knnLoss(SigmoidLoss):
    def __init__(self, cfg=None):
        super(knnLoss, self).__init__()

    def is_single(self):
        return False

    def loss(self, logits, knn_logits, targets, per_cls_weights, coeff, kwargs):
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        )
        loss = F.cross_entropy(logits, targets, weight, reduction="none")

        p = knn_logits / torch.sum(knn_logits, -1, keepdims=True)
        knn_loss = F.nll_loss(torch.clamp(torch.log(p), min=-100),
            targets, weight, reduction="none")

        loss = loss + torch.mul(loss, knn_loss * coeff)
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, knn_logits,
        targets, per_cls_weights, coeff, multihot_targets=False
    ):
        loss = self.loss(
            pred_logits, knn_logits, targets,
            per_cls_weights, coeff, multihot_targets)
        return loss


class knnFocalLikeLoss(SigmoidLoss):
    def __init__(self, cfg=None):
        super(knnFocalLikeLoss, self).__init__()

    def is_single(self):
        return False

    def loss(self, logits, knn_logits, targets, per_cls_weights, gamma, kwargs):
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        )
        loss = F.cross_entropy(logits, targets, weight, reduction="none")

        num_classes = logits.shape[1]
        targets = self.multi_hot(targets, num_classes)
        # modulator = (1 - p_t) ** gamma
        # below is a numerically stable version
        p = knn_logits / torch.sum(knn_logits, -1, keepdims=True)
        p_t = torch.sum(p * targets, -1)
        # a mask of p == 0
        modulator = torch.exp(gamma * torch.log1p(-1 * p_t))

        loss = loss * modulator
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, knn_logits,
        targets, per_cls_weights, coeff, multihot_targets=False
    ):
        loss = self.loss(
            pred_logits, knn_logits, targets,
            per_cls_weights, coeff, multihot_targets)
        return loss


LOSS = {
    "knn_focalstar": knnFocalLikeLoss,  # using (1-p)** gamma as weight for the whole instance
    "knn_reg": knnLoss,
    "softmax": SoftmaxLoss,
}


def build_loss(cfg):
    loss_name = cfg.SOLVER.LOSS
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not supported'
    loss_fn = LOSS[loss_name]
    if not loss_fn:
        return None
    else:
        return loss_fn(cfg)
