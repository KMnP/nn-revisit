#!/usr/bin/env python3

"""ImageNet dataset."""
import os
import torch
import torch.utils.data
import torchvision as tv

from ..transforms import get_transforms
from ...utils import logging
logger = logging.get_logger("nearest_neighbors")
# Data location
_DATA_DIR = "/datasets01/imagenet_full_size/061417"


class ImageNet(tv.datasets.ImageFolder):
    def __init__(self, cfg, split):
        super(ImageNet, self).__init__(
            root=os.path.join(_DATA_DIR, split),
            transform=get_transforms(split, cfg.DATA.CROPSIZE)
        )
        self.name = cfg.DATA.NAME
        self.cfg = cfg
        self._split = split

    def get_info(self):
        return len(self.imgs), self.get_class_num()

    def get_class_num(self):
        return len(self.class_to_idx)

    def __getitem__(self, index):
        # Load the image
        im, label = super(ImageNet, self).__getitem__(index)
        sample = {
            "image": im,
            "label": label,
            "id": index
        }
        return sample
