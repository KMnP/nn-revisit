#!/usr/bin/env python3

"""
ResNet models with knn variants.
Note: both models return probs instead of logits
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .build_vit_backbone import build_vit_models
from .linear_model import MLP
from .knn_utils import entropy
from ..utils import logging
logger = logging.get_logger("nearest_neighbors")
MODEL_ROOT=""
SIZE_MODEL_ROOT = ""


class ResNet50(nn.Module):
    """ResNet model."""

    def __init__(self, cfg, dstore=None):
        super(ResNet50, self).__init__()

        model_type = cfg.DATA.FEATURE
        model = self.get_model(model_type)
        # output shape: num_batch x 2048 x 7 x 7
        self.enc = nn.Sequential(*list(model.children())[:-2])
        self.load_size_ckpts(model_type)

        if cfg.MODEL.POOL == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif cfg.MODEL.POOL == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))

        if cfg.MODEL.FROZEN:
            for p in self.enc.parameters():
                p.requires_grad = False
        self.cfg = cfg
        self.dstore = dstore
        self.setup_head(cfg)

        self.fmap_pool = build_knn_fmappool(cfg)

    def get_model(self, model_type):
        if model_type == "imagenet_supervised":
            model = models.resnet50(pretrained=True)

        elif model_type == "imagenet_sup_rn101":
            model = models.resnet101(pretrained=True)  # 2048
        elif model_type == "imagenet_sup_rn152":
            model = models.resnet152(pretrained=True)  # 2048
        elif model_type == "imagenet_sup_rn34":
            model = models.resnet34(pretrained=True)  # 512
        elif model_type == "imagenet_sup_rn18":
            model = models.resnet18(pretrained=True)  # 512

        elif model_type == "random":
            model = models.resnet50(pretrained=False)

        elif "imagenet_sup" in model_type or "imagenet22k_sup" in model_type:
            # size experiments
            model = models.resnet50(pretrained=False)

        elif model_type == "imagenet21k_miil":
            model = models.resnet50(pretrained=False)
            model = self._load_imagenet22k_miil(model)

        elif model_type == "imagenet_barlowtwins":
            model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')

        elif model_type == 'dino_resnet50':
            model = torch.hub.load('facebookresearch/dino:main', model_type)

        elif model_type == "inat2021_supervised":
            checkpoint = torch.load(
                f"{MODEL_ROOT}/inat2021_supervised_large.pth.tar",
                map_location=torch.device('cpu')
            )
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 10000)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        elif model_type == 'inat2021_mini_supervised':
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 10000)
            checkpoint = torch.load(
                f"{MODEL_ROOT}/inat2021_supervised_mini.pth.tar",
                map_location=torch.device('cpu')
            )
            model.load_state_dict(checkpoint['state_dict'], strict=True)

        elif model_type == 'inat2021_mini_moco_v2':
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Identity()
            checkpoint = torch.load(
                f"{MODEL_ROOT}/inat2021_moco_v2_mini_1000_ep.pth.tar",
                map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            model.load_state_dict(state_dict, strict=True)

        elif model_type == 'imagenet_moco_v2':
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Identity()
            checkpoint = torch.load(
                f"{MODEL_ROOT}/imagenet_moco_v2_800ep_pretrain.pth.tar",
                map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            model.load_state_dict(state_dict, strict=True)

        elif model_type.startswith("mocov3_rn50"):
            moco_epoch = model_type.split("ep")[-1]
            checkpoint = torch.load(
                f"{SIZE_MODEL_ROOT}/mocov3_linear-{moco_epoch}ep.pth.tar",
                map_location="cpu")
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            model = models.resnet50()
            model.load_state_dict(state_dict, strict=False)

        elif model_type == 'imagenet_swav':
            # or could load from hub model
            model = torch.hub.load('facebookresearch/swav', 'resnet50')

        elif model_type == 'inat2021_mini_swav':
            # or could load from hub model
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Identity()
            state_dict = torch.load(
                f"{MODEL_ROOT}/inat2021_swav_mini_1000_ep.pth",
                map_location="cpu"
            )

            state_dict = {k.replace("module.", ""): v for k, v in state_dict['state_dict'].items()}
            for k in list(state_dict.keys()):
                if 'projection' in k or 'prototypes' in k:
                    del state_dict[k]

            model.load_state_dict(state_dict, strict=True)

        else:
            raise ValueError("model type not supported for resnet backbone")

        return model

    def load_size_ckpts(self, model_type):
        if model_type not in [
            "imagenet_sup0.75",
            "imagenet_sup0.5",
            "imagenet_sup0.25",
        ]:
            return

        f = os.path.join(SIZE_MODEL_ROOT, f"{model_type}.pyth")
        checkpoint = torch.load(f, map_location=torch.device("cpu"))
        state_dict = checkpoint['model_state']
        for k in list(state_dict.keys()):
            if k.startswith('module.enc'):
                # remove prefix
                state_dict[k[len("module.enc."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        self.enc.load_state_dict(state_dict, strict=True)
        logger.info("Loading pre-trained model weights for imageNet data size experiment")

    def _load_imagenet22k_miil(self, model):
        model_path = f"{SIZE_MODEL_ROOT}/resnet50_miil_21k.pth"
        state = torch.load(model_path, map_location='cpu')
        for key in model.state_dict():
            if 'num_batches_tracked' in key:
                continue
            p = model.state_dict()[key]
            if key in state['state_dict']:
                ip = state['state_dict'][key]
                if p.shape == ip.shape:
                    p.data.copy_(ip.data)  # Copy the data of parameters
                else:
                    print(
                        'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
            else:
                print('could not load layer: {}, not in checkpoint'.format(key))
        return model

    def setup_head(self, cfg):
        if self.cfg.MODEL.POOL == "convlite":
            out_dim = 128
        elif self.cfg.MODEL.POOL == "conv_pyramid":
            out_dim = 128
        elif cfg.DATA.FEATURE == "imagenet_sup_rn34" or cfg.DATA.FEATURE == "imagenet_sup_rn18":
            out_dim = 512
        else:
            out_dim = 2048
        self.head = MLP(
            input_dim=out_dim,
            mlp_dims=[out_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES],
            special_bias=True
        )

    def set_train(self):
        if self.cfg.MODEL.FROZEN:
            self.enc.eval()
            self.pool.eval()
            self.head.train()
        else:
            self.train()

    def forward(self, x, image_ids=None, return_feature=False, imagenet=False):
        if self.cfg.MODEL.FROZEN and self.enc.training:
            self.enc.eval()
        with torch.set_grad_enabled(self.enc.training):
            x, knn_x = self.get_features(x)

        if return_feature:
            return x, knn_x
        x = self.head(x)

        if imagenet:
            return x
        return {"probs": x}

    def get_features(self, x):
        """get a (batch_size, 2048) feature"""
        featmaps = self.enc(x)  # batch_size x 2048 x 7 x 7
        x = self.pool(featmaps)
        x = x.view(x.size(0), -1)

        knn_x = self.fmap_pool(featmaps)
        knn_x = knn_x.view(knn_x.size(0), -1)
        return x, knn_x


def build_knn_fmappool(cfg):
    if cfg.DSTORE.FEATUREMAP_POOL == "avg":
        fmap_pool = nn.AdaptiveAvgPool2d((1, 1))
    elif cfg.DSTORE.FEATUREMAP_POOL == "max":
        fmap_pool = nn.AdaptiveMaxPool2d((1, 1))
    else:
        raise ValueError("fmap_pool type {} is not supported".format(
            cfg.DSTORE.FEATUREMAP_POOL))
    return fmap_pool


class ResNet50JOINT(ResNet50):
    def __init__(self, cfg, dstore):
        super(ResNet50JOINT, self).__init__(cfg, dstore)
        # self.coeff = torch.nn.Parameter(torch.tensor(cfg.MODEL.KNN_LAMBDA))
        # add
        knn_model_type = cfg.DATA.DSTORE_FEATURE
        if len(knn_model_type) > 0:
            try:
                model = self.get_model(knn_model_type)
                # output shape: num_batch x 2048 x 7 x 7
                self.knn_enc = nn.Sequential(*list(model.children())[:-2])
            except ValueError as e:
                self.knn_enc, _ = build_vit_models(
                    knn_model_type, cfg.DATA.CROPSIZE)
            if cfg.MODEL.FROZEN:
                for p in self.knn_enc.parameters():
                    p.requires_grad = False
        else:
            self.knn_enc = None

        self.coeff = torch.tensor(cfg.MODEL.KNN_LAMBDA)
        self.fmap_pool = build_knn_fmappool(cfg)

        self.dstore = dstore
        self.dstore_return_probs = cfg.DSTORE.RETURN_PROBS

        self.imageid2knnx = {}
        self.use_cache = cfg.DSTORE.USE_CACHE
        self.activation = "softmax"

    def reset_cache(self):
        self.imageid2knnx = {}

    def setup_head(self, cfg):
        if cfg.DATA.FEATURE == "imagenet_sup_rn34" or cfg.DATA.FEATURE == "imagenet_sup_rn18":
            out_dim = 512
        else:
            out_dim = 2048

        if cfg.DSTORE.FEAT_MLP:
            self.knn_feathead = MLP(
                input_dim=2048, mlp_dims=[2048, 128], normalization=None,
                dropout=0, special_bias=False)
        else:
            self.knn_feathead = None

        if not cfg.DSTORE.RETURN_PROBS:
            if self.knn_feathead is not None:
                add_dim = 128
            else:
                add_dim = 2048

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
            input_dim=out_dim + add_dim,
            mlp_dims=[out_dim + add_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES],
            special_bias=True
        )

    def _get_features(self, x, image_ids):
        """get a (batch_size, 2048) feature"""
        if self.knn_enc is not None:
            featmaps = self.knn_enc(x)
        else:
            featmaps = self.enc(x)  # batch_size x 2048 x 7 x 7

        if image_ids is None:
            if len(featmaps.shape) > 2:
                knn_x = self.fmap_pool(featmaps)
                knn_x = knn_x.view(knn_x.size(0), -1)
            else:
                knn_x = featmaps

        elif str(image_ids[0]) in self.imageid2knnx:
            # print("use cache", len(image_ids))
            out = []
            for q_id in image_ids:
                out.append(self.imageid2knnx[str(q_id)])
            knn_x = torch.cat(out, 0)

        else:
            if len(featmaps.shape) > 2:
                knn_x = self.fmap_pool(featmaps)
                knn_x = knn_x.view(knn_x.size(0), -1)
            else:
                knn_x = featmaps
            # # add back to the cache TODO: add one method to save all features
            if self.use_cache:
                for i in range(len(image_ids)):
                    # (1, num_classes)
                    self.imageid2knnx[str(image_ids[i])] = knn_x[i, :].unsqueeze(0)

        if self.knn_feathead is not None:
            knn_x = self.knn_feathead(knn_x)

        if self.knn_enc is not None:
            featmaps = self.enc(x)
        x = self.pool(featmaps)
        x = x.view(x.size(0), -1)
        return x, knn_x

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
            if self.knn_enc is not None and self.knn_enc.training:
                self.knn_enc.eval()

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
                knn_probs = F.softmax(self.knn_head(knn_x))
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


def combine_knn_and_linear_probs(
    knn_logits, model_logits, coeff, activation="softmax"):
    if activation == "sigmoid":
        knn_p = torch.sigmoid(knn_logits)
        model_p = torch.sigmoid(model_logits)
    elif activation == "softmax":
        knn_p = knn_logits / torch.sum(knn_logits, -1, keepdims=True)
        model_p = F.softmax(model_logits, dim=1)
    return knn_p * coeff + model_p * (1 - coeff)
