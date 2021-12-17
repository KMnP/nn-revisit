#!/usr/bin/env python3

"""
ViT models with knn variants.
Note: both models return probs instead of logits
"""
import torch
import torch.nn as nn

from .resnet import combine_knn_and_linear_probs
from .build_vit_backbone import build_vit_models
from .linear_model import MLP
from ..utils import logging
logger = logging.get_logger("nearest_neighbors")


class ViT(nn.Module):
    """ViT model."""

    def __init__(self, cfg, dstore=None):
        super(ViT, self).__init__()

        self.enc, self.feat_dim = build_vit_models(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE)

        if cfg.MODEL.FROZEN:
            for p in self.enc.parameters():
                p.requires_grad = False
        self.cfg = cfg
        self.dstore = dstore
        self.setup_head(cfg)

    def setup_head(self, cfg):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES],
            special_bias=True
        )

    def forward(self, x, image_ids=None, return_feature=False, imagenet=False):
        if self.cfg.MODEL.FROZEN and self.enc.training:
            self.enc.eval()
        with torch.set_grad_enabled(self.enc.training):
            x = self.get_features(x)

        if return_feature:
            return x, x
        x = self.head(x)

        if imagenet:
            return x
        # convert to softmax
        # x = F.softmax(x)
        return {"probs": x}

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x


class ViTJOINT(ViT):
    # may not need this since jointly train probs is not good enough
    def __init__(self, cfg, dstore):
        super(ViTJOINT, self).__init__(cfg, dstore)
        # self.coeff = torch.nn.Parameter(torch.tensor(cfg.MODEL.KNN_LAMBDA))
        self.coeff = torch.tensor(cfg.MODEL.KNN_LAMBDA)
        self.dstore = dstore
        self.dstore_return_probs = cfg.DSTORE.RETURN_PROBS
        self.imageid2knnx = {}
        self.use_cache = cfg.DSTORE.USE_CACHE
        self.activation = "softmax"

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

    def _get_features(self, x, image_ids):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim

        if image_ids is None:
            knn_x = x.clone()
        elif str(image_ids[0]) in self.imageid2knnx:
            # print("use cache", len(image_ids))
            out = []
            for q_id in image_ids:
                out.append(self.imageid2knnx[str(q_id)])
            knn_x = torch.cat(out, 0)

        else:
            knn_x = x.clone()
            # add back to the cache
            if self.use_cache:
                for i in range(len(image_ids)):
                    # (1, num_classes)
                    self.imageid2knnx[str(image_ids[i])] = knn_x[i, :].unsqueeze(0)

        if self.knn_feathead is not None:
            knn_x = self.knn_feathead(knn_x)

        return x, knn_x

    def reset_cache(self):
        self.imageid2knnx = {}

    def get_features(self, x, image_ids=None):
        return self._get_features(x, image_ids)

    def get_base_probs(self, x):
        # return F.softmax(self.head(x))
        return self.head(x)

    def get_knn_probs(self, knn_x, image_ids):
        if image_ids is not None and isinstance(image_ids[0], str) and not image_ids[0].startswith("tensor"):
            # for test and val images
            return self.dstore(knn_x, None)
        return self.dstore(knn_x, image_ids)

    def forward(self, x, image_ids=None, return_feature=False):
        if self.cfg.MODEL.FROZEN and self.enc.training:
            self.enc.eval()

        with torch.set_grad_enabled(self.enc.training):
            x, knn_x = self.get_features(x, image_ids)

        if return_feature:
            return x, knn_x

        if self.dstore_return_probs:
            base_probs = self.get_base_probs(x)
            knn_probs = self.get_knn_probs(knn_x, image_ids)
            probs = combine_knn_and_linear_probs(
                knn_probs, base_probs, self.coeff, self.activation)
        else:
            knn_x = self.dstore(knn_x, image_ids)
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
            "features": knn_x
        }
