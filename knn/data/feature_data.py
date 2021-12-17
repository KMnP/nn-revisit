#!/usr/bin/env python3
"""a dataset class that handle numpy arrays as input (for NEWT)"""

import numpy as np
import torch

from collections import Counter
from ..utils import logging
from ..utils.io_utils import get_feature_df
logger = logging.get_logger("nearest_neighbors")


class featureData(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        assert split in {
            "train",
            "val",
        }, "Split '{}' not supported for feature data".format(split)
        logger.info("\tConstructing {} dataset {}...".format(
            cfg.DATA.NAME, split))
        self.cfg = cfg
        if split == "val":
            self._split = "test"
        else:
            self._split = split
        self.name = cfg.DATA.NAME
        self._construct_imdb(cfg)

    def _construct_imdb(self, cfg):
        """Constructs the imdb."""
        feature_df = get_feature_df(cfg)
        _df = feature_df[feature_df["name"] == cfg.DATA.NAME].iloc[0]

        # list of labels in int
        self.targets = list(_df[f"y_{self._split}"].astype(np.int))
        # num_data x feature_dim
        self.feature_array = _df[f"X_{self._split}"].astype(np.float32)
        self._class_ids = sorted(list(set(self.targets)))

        logger.info("\tNumber of images: {}".format(len(self.targets)))
        logger.info("\tNumber of classes: {}".format(self.get_class_num()))
        if self.get_class_num() != cfg.DATA.NUMBER_CLASSES:
            raise ValueError("please specify correct number of classes in config, should be {self.get_class_num()} for {self.name}, got {cfg.DATA.NUMBER_CLASSES}")  # noqa

    def get_class_num(self) -> int:
        return len(self._class_ids)

    def get_info(self):
        num_imgs = self.feature_array.shape[0]
        return num_imgs, self.get_class_num()

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if self._split != "train":
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        label = self.targets[index]
        input_feature = self.feature_array[index, :]  # (feature_dim, )
        if self._split == "train":
            index = index
        else:
            index = f"{self._split}{index}"
        sample = {
            "image": input_feature,
            "label": label,
            "id": index
        }
        return sample

    def __len__(self):
        return len(self.targets)
