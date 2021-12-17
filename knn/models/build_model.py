#!/usr/bin/env python3
"""
Model construction functions.
"""
import torch

from .knn_dstore import KNNtorch

from .linear_model import LinearModel, LinearJoint
from .resnet import ResNet50, ResNet50JOINT
from .vit import ViT, ViTJOINT
from ..utils import logging
logger = logging.get_logger("nearest_neighbors")
# Supported model types
_MODEL_TYPES = {
    # resnet
    "base": ResNet50,
    "joint": ResNet50JOINT,
    # vit
    "vit_base": ViT,
    "vit_joint": ViTJOINT,
    # legacy issues
    "dino_base": ViT,
    "dino_joint": ViTJOINT,
    # swin-transformer
    # dstore types
    "linear_base": LinearModel,
    "linear_joint": LinearJoint
}


def build_dstore(cfg, cur_device, train_loader=None):
    logger.info("Constructing datastores...")
    logger.info(f"Setting up dstore using {cfg.DSTORE.TYPE}")
    dstore = KNNtorch(cfg, cur_device)

    if cfg.DSTORE.TOPK_TYPE is not None:
        num_imgs, num_clses = train_loader.dataset.get_info()
        if cfg.DSTORE.TOPK_TYPE == "max":
            dstore.update_k(num_imgs)
        elif cfg.DSTORE.TOPK_TYPE == "avg_cls_count":
            dstore.update_k(int(num_imgs / num_clses))
    return dstore


def build_model(cfg, train_loader=None):
    """
    build model here
    """
    assert (
        cfg.MODEL.TYPE in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.MODEL.TYPE)
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    dstore = build_dstore(cfg, get_current_device(), train_loader)

    train_type = cfg.MODEL.TYPE
    if cfg.DATA.FEATURE == "dino_resnet50":
        train_type = train_type.replace("vit_", "")
    elif"swin" in cfg.DATA.FEATURE or "dino" in cfg.DATA.FEATURE or "sup_vit" in cfg.DATA.FEATURE or "sup_xcit" in cfg.DATA.FEATURE or "mocov3_vit" in cfg.DATA.FEATURE:
        if not train_type.startswith("vit_"):
            train_type = "vit_" + train_type
    model = _MODEL_TYPES[train_type](cfg, dstore)

    log_model_info(model)
    model, device = load_model_to_device(model, cfg)
    logger.info(f"Device used for model: {device}")

    return model, device


def log_model_info(model):
    """Logs model info"""
    logger.info(f"Classification Model:\n{model}")
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))


def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
    else:
        cur_device = torch.device('cpu')
    return cur_device


def load_model_to_device(model, cfg):
    cur_device = get_current_device()
    if torch.cuda.is_available():
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
        # Use multi-process data parallel model in the multi-gpu setting
        if cfg.NUM_GPUS > 1:
            # Make model replica operate on the current device
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device,
                find_unused_parameters=True,
            )
    else:
        model = model.to(cur_device)
    return model, cur_device
