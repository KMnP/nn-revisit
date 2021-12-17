#!/usr/bin/env python3
"""
Modified from: fbcode/multimo/models/encoders/mlp.py
"""
import math
import torch
import torch.nn.functional as F

from torch import nn
from typing import List, Type

from ..utils import logging
logger = logging.get_logger("nearest_neighbors")


def combine_knn_and_linear_probs(
    knn_logits, model_logits, coeff, activation="softmax"):
    if activation == "sigmoid":
        knn_p = torch.sigmoid(knn_logits)
        model_p = torch.sigmoid(model_logits)
    elif activation == "softmax":
        knn_p = knn_logits / torch.sum(knn_logits, -1, keepdims=True)
        model_p = F.softmax(model_logits, dim=1)
    return knn_p * coeff + model_p * (1 - coeff)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dims: List[int],
        dropout: float = 0.1,
        nonlinearity: Type[nn.Module] = nn.ReLU,
        normalization: Type[nn.Module] = nn.BatchNorm1d,  # nn.LayerNorm,
        special_bias: bool = False,
        add_bn_first: bool = False,
    ):
        super(MLP, self).__init__()
        projection_prev_dim = input_dim
        projection_modulelist = []
        last_dim = mlp_dims[-1]
        mlp_dims = mlp_dims[:-1]

        if add_bn_first:
            if normalization is not None:
                projection_modulelist.append(normalization(projection_prev_dim))
            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))

        for idx, mlp_dim in enumerate(mlp_dims):
            fc_layer = nn.Linear(projection_prev_dim, mlp_dim)
            nn.init.kaiming_normal_(fc_layer.weight, a=0, mode='fan_out')
            projection_modulelist.append(fc_layer)
            projection_modulelist.append(nonlinearity())

            if normalization is not None:
                projection_modulelist.append(normalization(mlp_dim))

            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))
            projection_prev_dim = mlp_dim

        self.projection = nn.Sequential(*projection_modulelist)
        self.last_layer = nn.Linear(projection_prev_dim, last_dim)
        nn.init.kaiming_normal_(self.last_layer.weight, a=0, mode='fan_out')
        if special_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.last_layer.bias, bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input_arguments:
            @x: torch.FloatTensor
        """
        x = self.projection(x)
        x = self.last_layer(x)
        return x


class LinearModel(nn.Module):
    def __init__(self, cfg, dstore=None):
        super(LinearModel, self).__init__()
        self.cfg = cfg
        self.dstore = dstore
        self.feat_dim = 2048  # TODO: only works for resnet 50
        self.setup_head(cfg)

    def setup_head(self, cfg):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES],
            special_bias=True
        )

    def forward(self, x, image_ids=None, return_feature=False, imagenet=False):

        if return_feature:
            return x, x
        x = self.head(x)

        return {"probs": x}


class LinearJoint(LinearModel):
    """
    during inference time, this model aggregate both probabilitics from MLP and knn for final results
    """
    def __init__(self, cfg, dstore=None):
        super(LinearJoint, self).__init__(cfg, dstore)
        self.coeff = torch.tensor(cfg.MODEL.KNN_LAMBDA)
        self.dstore = dstore
        self.dstore_return_probs = cfg.DSTORE.RETURN_PROBS
        if cfg.SOLVER.LOSS == "softmax" or cfg.SOLVER.LOSS == "knn_reg" or cfg.SOLVER.LOSS == "knn_reg_test" or cfg.SOLVER.LOSS == "knn_focalstar":
            self.activation = "softmax"
        else:
            self.activation = "sigmoid"

    def setup_head(self, cfg):
        if cfg.DSTORE.FEAT_MLP:
            self.knn_feathead = MLP(
                input_dim=self.feat_dim, mlp_dims=[self.feat_dim, 128], normalization=None,
                dropout=0, special_bias=False)
        else:
            self.knn_feathead = None

        if not cfg.DSTORE.RETURN_PROBS:
            if self.knn_feathead is not None:
                add_dim = 128
            else:
                add_dim = self.feat_dim

            if cfg.DSTORE.SEP_HEAD:
                self.knn_head = MLP(
                    input_dim=add_dim, mlp_dims=[cfg.DATA.NUMBER_CLASSES],
                    special_bias=True
                )
                add_dim = 0
            else:
                self.knn_head = None
        else:
            add_dim = 0
        self.head = MLP(
            input_dim=self.feat_dim + add_dim,
            mlp_dims=[self.feat_dim + add_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES],
            special_bias=True
        )

    def get_base_probs(self, x):
        # return F.softmax(self.head(x))
        return self.head(x)

    def get_knn_probs(self, knn_x, image_ids):
        if image_ids is not None and isinstance(image_ids[0], str) and not image_ids[0].startswith("tensor"):
            # for test and val images
            return self.dstore(knn_x, None)
        return self.dstore(knn_x, image_ids)

    def forward(self, x, image_ids=None, return_feature=False):
        if return_feature:
            return x, x

        if self.dstore_return_probs:
            base_probs = self.get_base_probs(x)
            knn_probs = self.get_knn_probs(x, image_ids)
            probs = combine_knn_and_linear_probs(
                knn_probs, base_probs, self.coeff, self.activation)
        else:
            knn_x = self.dstore(x, image_ids)
            if self.knn_head is not None:
                knn_probs = self.knn_head(knn_x)
                # knn_probs = F.softmax(self.knn_head(knn_x))
                base_probs = self.get_base_probs(x)
                probs = combine_knn_and_linear_probs(
                    knn_probs, base_probs, self.coeff, self.activation)
            else:
                probs = self.get_base_probs(torch.cat([x, knn_x], 1))
                base_probs = None
                knn_probs = None

        return {
            "probs": probs, "base_probs": base_probs,
            "knn_probs": knn_probs,
            "features": x
        }
