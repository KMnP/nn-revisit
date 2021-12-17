import numpy as np
import torch
import os
from .swin_transformer import SwinTransformer
from .swin_mlp import SwinMLP
from .vit_backbones import VisionTransformer
from .vit_moco import vit_base, vit_small
MODEL_ROOT = ""  # TODO: MOVE TO CONFIG
MODEL_ZOO = {
    "swint_imagenet": "swin_tiny_patch4_window7_224.pth",
    "swint_imagenet_ssl": "moby_swin_t_300ep_pretrained.pth",
    "swins_imagenet": "swin_small_patch4_window7_224.pth",
    "swinb_imagenet_224": "swin_base_patch4_window7_224.pth",
    "swinb_imagenet_384": "swin_base_patch4_window12_384.pth",
    "swinb_imagenet22k_224":  "swin_base_patch4_window7_224_22k.pth",
    "swinb_imagenet22k_384": "swin_base_patch4_window12_384_22k.pth",
    "swinl_imagenet22k_224": "swin_large_patch4_window7_224_22k.pth",
    "sup_vitb8": "ViT-B_8.npz",
    "sup_vitb16_224": "ViT-B_16-224.npz",
    "sup_vitb16": "ViT-B_16.npz",
    "sup_vitl16_224": "ViT-L_16-224.npz",
    "sup_vitl16": "ViT-L_16.npz",
    "sup_vitb16_imagenet22k": "imagenet22k_ViT-B_16.npz",
    "sup_vitl16_imagenet22k": "imagenet22k_ViT-L_16.npz",
}


def build_mocov3_model(model_type):
    if model_type == "mocov3_vits":
        model = vit_small()
        out_dim = 384
        ckpt = f"{MODEL_ROOT}/mocov3_linear-vit-s-300ep.pth.tar"
    elif model_type == "mocov3_vitb":
        model = vit_base()
        out_dim = 768
        ckpt = f"{MODEL_ROOT}/mocov3_linear-vit-b-300ep.pth.tar"

    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.'):
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.head = torch.nn.Identity()
    return model, out_dim


def build_xcit_model(model_type):
    # "dino_xcit_small_12_p16": 384
    # "sup_xcit_small_12_p16": 384
    # "sup_xcit_small_12_p16_dist": 384

    # "dino_xcit_small_12_p8": 384
    # "sup_xcit_small_12_p8": 384
    # "sup_xcit_small_12_p8_dist": 384

    # "dino_xcit_medium_24_p16": 512
    # "sup_xcit_medium_24_p16": 512
    # "sup_xcit_medium_24_p16_dist": 512

    # "dino_xcit_medium_24_p8": 512
    # "sup_xcit_medium_24_p8": 512
    # "sup_xcit_medium_24_p8_dist": 512

    #
    if "xcit_small_12_p16" in model_type:
        model =torch.hub.load(
            'facebookresearch/dino:main', 'dino_xcit_small_12_p16')
        if model_type.startswith("sup") and model_type.endswith("dist"):
            ckpts = torch.load(
                f"{MODEL_ROOT}/xcit_small_12_p16_224_dist.pth",
                map_location=torch.device('cpu')
            )
            model.load_state_dict(ckpts['model'], strict=False)
        elif model_type.startswith("sup"):
            ckpts = torch.load(
                f"{MODEL_ROOT}/xcit_small_12_p16_224.pth",
                map_location=torch.device('cpu')
            )
            model.load_state_dict(ckpts['model'], strict=False)
        return model, 384

    elif "xcit_medium_24_p16" in model_type:
        model = torch.hub.load(
            'facebookresearch/dino:main', 'dino_xcit_medium_24_p16')
        if model_type.startswith("sup") and model_type.endswith("dist"):
            ckpts = torch.load(
                f"{MODEL_ROOT}/xcit_medium_24_p16_224_dist.pth",
                map_location=torch.device('cpu')
            )
            model.load_state_dict(ckpts['model'], strict=False)
        elif model_type.startswith("sup"):
            ckpts = torch.load(
                f"{MODEL_ROOT}/xcit_medium_24_p16_224.pth",
                map_location=torch.device('cpu')
            )
            model.load_state_dict(ckpts['model'], strict=False)
        return model, 512

    elif "xcit_small_12_p8" in model_type:
        model = torch.hub.load(
            'facebookresearch/dino:main', 'dino_xcit_small_12_p8')
        if model_type.startswith("sup") and model_type.endswith("dist"):
            ckpts = torch.load(
                f"{MODEL_ROOT}/xcit_small_12_p8_224_dist.pth",
                map_location=torch.device('cpu')
            )
            model.load_state_dict(ckpts['model'], strict=False)
        elif model_type.startswith("sup"):
            ckpts = torch.load(
                f"{MODEL_ROOT}/xcit_small_12_p8_224.pth",
                map_location=torch.device('cpu')
            )
            model.load_state_dict(ckpts['model'], strict=False)
        return model, 384

    elif "xcit_medium_24_p8" in model_type:
        model = torch.hub.load(
            'facebookresearch/dino:main', 'dino_xcit_medium_24_p8')
        if model_type.startswith("sup") and model_type.endswith("dist"):
            ckpts = torch.load(
                f"{MODEL_ROOT}/xcit_medium_24_p8_224_dist.pth",
                map_location=torch.device('cpu')
            )
            model.load_state_dict(ckpts['model'], strict=False)
        elif model_type.startswith("sup"):
            ckpts = torch.load(
                f"{MODEL_ROOT}/xcit_medium_24_p8_224.pth",
                map_location=torch.device('cpu')
            )
            model.load_state_dict(ckpts['model'], strict=False)
        return model, 512
    else:
        raise ValueError(f"{model_type} is not supported for xcit")


def build_swin_model(model_type, crop_size):
    if model_type == "swint_imagenet":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,  # setting to a negative value will make head as identity
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swint_imagenet_ssl":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4

    elif model_type == "swins_imagenet":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.3,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swinb_imagenet_224":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet_384":
        model = SwinTransformer(
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4

    elif model_type == "swinb_imagenet22k_224":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet22k_384":
        model = SwinTransformer(
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinl_imagenet22k_224":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 192
        num_layers = 4

    feat_dim = int(embed_dim * 2 ** (num_layers - 1))
    # load checkpoint
    model_w = os.path.join(MODEL_ROOT, MODEL_ZOO[model_type])
    checkpoint = torch.load(model_w, map_location='cpu')
    state_dict = checkpoint['model']

    if crop_size == 448:
        for k in list(state_dict.keys()):
            if "attn_mask" not in k:
                # remove prefix
                state_dict[k] = state_dict[k]
            # delete renamed or unused k
            else:
                del state_dict[k]

    # rename some keys for ssl models
    if model_type.endswith("ssl"):
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder.'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    return model, feat_dim


def build_dino_models(model_type):
    m2featdim = {
        'dino_vits16': 384,
        'dino_vits8': 384,
        'dino_vitb16': 768,
        'dino_vitb8': 768,
        # 'dino_xcit_small_12_p16',
        # 'dino_xcit_small_12_p8',
        # 'dino_xcit_medium_24_p16',
        # 'dino_xcit_medium_24_p8',
        'dino_resnet50': 2048,
    }

    if model_type not in m2featdim:
        raise ValueError("model type is not supported for Dino")

    model = torch.hub.load('facebookresearch/dino:main', model_type)
    return model, m2featdim[model_type]


def build_vit_sup_models(model_type, crop_size):
    # image size is the size of actual image
    m2featdim = {
        "sup_vitb16_224": 768,
        "sup_vitb16": 768,
        "sup_vitl16_224": 1024,
        "sup_vitl16": 1024,
        "sup_vitb16_imagenet22k": 768,
        "sup_vitl16_imagenet22k": 1024,
    }
    model = VisionTransformer(model_type, crop_size, num_classes=-1)
    model.load_from(np.load(os.path.join(MODEL_ROOT, MODEL_ZOO[model_type])))
    return model, m2featdim[model_type]


def build_vit_models(model_type, crop_size):
    if model_type.startswith("dino"):
        if "dino_xcit" in model_type:
            return build_xcit_model(model_type)
        return build_dino_models(model_type)
    elif model_type.startswith("swin"):
        return build_swin_model(model_type, crop_size)
    elif model_type.startswith("sup_vit"):
        return build_vit_sup_models(model_type, crop_size)
    elif "xcit" in model_type:
        return build_xcit_model(model_type)
    elif "mocov3" in model_type:
        return build_mocov3_model(model_type)
