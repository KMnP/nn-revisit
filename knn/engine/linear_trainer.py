#!/usr/bin/env python3
"""
a trainer class for linear evaluation using continous knn
"""
import numpy as np
import torch
import torch.nn as nn
import os

from fvcore.common.config import CfgNode

from ..engine.evaluator import Evaluator
from ..utils import logging
from .trainer import Trainer
logger = logging.get_logger("nearest_neighbors")
FEAT_DIR=""  # TODO: move it to config


class LinearTrainer(Trainer):
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        super(LinearTrainer, self).__init__(cfg, model, evaluator, device)

    def dstore_checkup(self, train_loader, val_loader, epoch):
        if epoch is None:
            return
        num_imgs, num_clses = train_loader.dataset.get_info()
        self.meank = int(num_imgs / num_clses)
        self.maxk = num_clses
        self.model.eval()

        if "conv" in self.cfg.DSTORE.FEATUREMAP_POOL:
            # update dstore regardless of the epoch
            self.update_dstore(train_loader, epoch)
        elif epoch == 0:  # only need to update once for linear eval with max/avg pool
            self.update_dstore(train_loader, epoch)
            if self.cfg.DSTORE.RETURN_PROBS:
                # tune knn using seperate validation data only
                self.tune_knn(val_loader)

    def update_dstore(self, train_loader, epoch):
        logger.info("Updating dstore index")

        # check if there is a saved features: for all linear exp, and finetune experiment with the first epoch
        r = FEAT_DIR

        if self.cfg.DATA.CROPSIZE == 224:
            postfix = ""
        else:
            postfix = "_448"
        outfile = os.path.join(r, "{}_{}_{}{}.pth".format(
            self.cfg.DATA.FEATURE, self.cfg.DATA.NAME,
            self.cfg.DSTORE.FEATUREMAP_POOL, postfix
        ))
        if os.path.exists(outfile) and "conv" not in self.cfg.DSTORE.FEATUREMAP_POOL:
            logger.info(
                f"loading previously saved train feature from {outfile}")
            data = torch.load(outfile, map_location=torch.device('cpu'))
            t_features = data["train_features"]
            train_labels = data["train_labels"]  # (num_train, 1)
            image_ids = data["image_ids"]  # (num_train, 1)
        else:
            t_features, t_labels, image_ids, t_base_features = self._generate_train_features(
                train_loader)
            self.compute_nmi(t_features, t_labels, "knn")
            self.compute_nmi(t_base_features, t_labels, "base")

            train_labels = np.array(t_labels)[:, np.newaxis]

            if "conv" not in self.cfg.DSTORE.FEATUREMAP_POOL:
                torch.save(
                    {
                        "train_features": t_features,
                        "train_labels": train_labels,
                        "image_ids": image_ids,
                    },
                    outfile
                )
                logger.info(f"saving the train feature to {outfile}")

        self.model.dstore.update_index(
            t_features,
            train_labels,
            image_ids,
        )

        logger.info("..updated index with train features size: {}".format(
            t_features.shape))
