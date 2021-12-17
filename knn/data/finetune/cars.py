#!/usr/bin/env python3

"""stanford-cars dataset."""

import os
import torch
import torch.utils.data
import torchvision as tv
import numpy as np
from collections import Counter

from ..transforms import get_transforms
from ...utils import logging
from ...utils.io_utils import read_json
logger = logging.get_logger("nearest_neighbors")
_DATA_DIR = ""  # need to update this
FOLDER = "Stanford-cars"


class CarsDataset(torch.utils.data.Dataset):
    """stanford-cars dataset."""

    def __init__(self, cfg, split):
        assert split in {
            "trainval",
            "train",
            "val",
            "test"
        }, "Split '{}' not supported for stanford-cars dataset".format(split)
        logger.info("Constructing stanford-cars dataset {}...".format(split))
        self.cfg = cfg
        self._split = split
        self.name = cfg.DATA.NAME
        self._construct_imdb(cfg)
        self.transform = get_transforms(self._split, cfg.DATA.CROPSIZE)

    def _construct_imdb(self, cfg):
        """Constructs the imdb."""
        anno_path = os.path.join(
            _DATA_DIR, FOLDER, "{}.json".format(self._split))
        img_dir = _DATA_DIR
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno = read_json(anno_path)
        # Map class ids to contiguous ids
        self._class_ids = sorted(list(set(anno.values())))
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image db
        self._imdb = []
        for img_name, cls_id in anno.items():
            cont_id = self._class_id_cont_id[cls_id]
            im_path = os.path.join(img_dir, img_name)
            self._imdb.append({"im_path": im_path, "class": cont_id})

        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))

    def get_info(self):
        num_imgs = len(self._imdb)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return len(self._class_ids)

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
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
        # Load the image
        im = tv.datasets.folder.default_loader(self._imdb[index]["im_path"])
        label = self._imdb[index]["class"]
        im = self.transform(im)
        if self._split == "train":
            index = index
        else:
            index = f"{self._split}{index}"
        sample = {
            "image": im,
            "label": label,
            "id": index
        }
        return sample

    def __len__(self):
        return len(self._imdb)
