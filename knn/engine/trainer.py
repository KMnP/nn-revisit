#!/usr/bin/env python3
"""
a trainer class for continous knn
"""
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import os

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer
from numpy import linalg as LA

from ..engine.eval.singlelabel import accuracy

from ..engine.evaluator import Evaluator
from ..engine.eval import cluster
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import distributed as du
from ..utils import logging
logger = logging.get_logger("nearest_neighbors")
FEAT_DIR=""  # TODO: move it to config


class Trainer():
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
        self.cfg = cfg
        self.model = model
        self.device = device

        # solver related
        logger.info("\tSetting up the otpmizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        if self.cfg.DSTORE.LOSS and "joint" in self.cfg.MODEL.TYPE:
            # compute knn-loss to add to the base loss for joint model only
            self.knn_loss_scale = torch.tensor(
                self.cfg.MODEL.KNN_LAMBDA,
                dtype=torch.float,
                device=self.device
            )
            if self.cfg.SOLVER.LOSS != "knn_reg" and self.cfg.SOLVER.LOSS != "knn_focal" and self.cfg.SOLVER.LOSS != "knn_base_focal" and self.cfg.SOLVER.LOSS != "knn_reg_test" and self.cfg.SOLVER.LOSS != "knn_focalstar":
                self.knn_criterion = build_loss(self.cfg)
            else:
                self.knn_criterion = None
        else:
            self.knn_criterion = None

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

    def compute_nmi(self, train_features, train_targets, prefix):
        # see if need to normalize feature
        if prefix == "knn":
            train_mean = train_features.mean(0)
            norm_type = self.cfg.DSTORE.NORM_TYPE
            if norm_type is not None and norm_type.startswith("L2"):
                if norm_type == "L2NC":
                    # norm_types: L2NC, center + l2 norm
                    train_features = train_features - train_mean
                train_features = train_features / LA.norm(
                    train_features, 2, 1)[:, None]

        nmi, a_nmi, v_nmi = cluster.nmi(
            train_targets, train_features, self.cfg.DATA.NUMBER_CLASSES)
        logger.info(
            "[{}] Clutering nmi: {:.4f}, adjusted nmi: {:.4f}, v: {:.4f}".format(
                prefix, nmi, a_nmi, v_nmi
            ))

    def forward_one_batch(self, inputs, targets, image_ids, is_train, epoch=None):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: train: logits; eval: probs
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )
        # image_ids = image_ids.to(self.device)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # zero the parameter gradients
        # self.optimizer.zero_grad()
        loss_dict = {}

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs, image_ids)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs["probs"].shape, targets.shape))

            if self.knn_criterion is None:
                # for base model, or knn model, or joint model with jointly optimized probs
                if self.cls_criterion.is_single():
                    loss = self.cls_criterion(
                        outputs["probs"], targets, self.cls_weights)
                    loss_dict["main"] = loss
                else:
                    loss = self.cls_criterion(
                        outputs["base_probs"], outputs["knn_probs"],
                        targets, self.cls_weights,
                        self.knn_loss_scale
                    )
                    loss_dict["main"] = loss

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss_dict["main"].backward()
            self.optimizer.step()

        del inputs
        return loss_dict, outputs

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        image_ids = data["id"]
        return inputs, labels, image_ids

    @torch.no_grad()
    def dstore_checkup(self, train_loader, val_loader, epoch):
        num_imgs, num_clses = train_loader.dataset.get_info()
        self.meank = int(num_imgs / num_clses)
        self.maxk = num_clses
        self.model.eval()
        if epoch == 0:
            # use previously saved features
            self.update_dstore_saved(train_loader, epoch)
        else:
            self.update_dstore(train_loader, epoch)

        if self.cfg.DSTORE.RETURN_PROBS:
            # if self.cfg.DSTORE.USE_CACHE and epoch == 0:
            # if epoch == 0:
                # tune knn using seperate validation data before first epoch
            self.tune_knn(val_loader)

    @torch.no_grad()
    def tune_knn(self, val_loader):
        # tune knn using smaller training set
        logger.info("Tuning temperature, k using val set")
        total_val_features = []
        total_val_targets = []

        with torch.no_grad():
            for _, input_data in enumerate(val_loader):
                X, targets, _ = self.get_input(input_data)
                X = X.to(self.device, non_blocking=True)
                _, knn_x = self.model(X, image_ids=None, return_feature=True)  # (batchsize, num_cls)
                total_val_features.append(knn_x)
                total_val_targets.extend(list(targets.numpy()))

        total_val_features = torch.cat(total_val_features, dim=0)

        # tune
        best_T, best_k, = None, None
        best_acc = -1

        if self.cfg.DSTORE.TEMP_RANGE == "full":
            temp_ranges =[0.0001, 0.001, 0.01, 0.1, 1, 10] + \
                [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09] + \
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            temp_ranges = [0.0001, 0.001, 0.01, 0.1, 1, 10]

        for T in temp_ranges:
            for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, self.meank, self.maxk]:
                if k > self.maxk:
                    continue
                self.model.dstore.update_params(k, T)
                probs = self.model.dstore.get_knn_prob(total_val_features)

                acc, _ = accuracy(probs.cpu().numpy(), total_val_targets)
                # print(k, acc)
                if acc > best_acc:
                    best_acc = acc
                    best_T, best_k = T, k

        # update:
        logger.info("Finish tuning dstore parameters: {}, {}, {}".format(
            best_k, best_T, -1))
        self.model.dstore.update_params(best_k, best_T)

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        num_warmup = min(5, log_interval - 1, total_epoch - 1)
        losses = AverageMeter('Loss', ':.4e')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        # first initialize dstore
        if "base" not in self.cfg.MODEL.TYPE:
            self.model.eval()
            self.dstore_checkup(train_loader, val_loader, 0)
            if not self.cfg.MODEL.FROZEN:
            # update val and test knn features in self.model
            # save all the knn_x in self.model?
                # self._generate_train_features(train_loader)
                self._generate_train_features(val_loader)
                self._generate_train_features(test_loader)
                # print([*self.model.imageid2knnx.keys()])

        for epoch in range(total_epoch):
            losses.reset()

            # set samplers
            if self.cfg.NUM_GPUS > 1:
                train_loader.sampler.set_epoch(epoch)
            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            # if self.cfg.NUM_GPUS > 1:
            #     self.model.module.set_train()
            # else:
            #     self.model.set_train()
            self.model.train()
            start_time = time.time()
            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break
                if idx == num_warmup:
                    start_time = time.time()
                X, targets, image_ids = self.get_input(input_data)

                ep_indicator = idx if self.cfg.DSTORE.ALTERNATE_LOSS else None

                # since we sample the training data as dstore, then we do not need to filter for id
                if self.cfg.NUM_GPUS == 1:
                    train_loss, _ = self.forward_one_batch(
                        X, targets, image_ids, True, ep_indicator)
                else:
                    train_loss, _ = self.forward_one_batch(
                        X, targets, None, True, ep_indicator)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss["main"].item(), X.shape[0])

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    if "knn" in train_loss:
                        optional_loss_str = "base: {:.4f}, knn: {:.4f}".format(
                            train_loss["base"], train_loss["knn"])
                    else:
                        optional_loss_str = ""
                    duration = time.time() - start_time
                    seconds_per_batch = duration / (idx + 1 - num_warmup)
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss["main"]
                        )
                        + optional_loss_str
                        + "{:.4f} s / batch. ETA={}, ".format(  # noqa
                            seconds_per_batch,
                            str(
                                datetime.timedelta(
                                    seconds=int(seconds_per_batch * (total_data - num_warmup) - duration)  # noqa
                                )
                            ),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: average train loss: {:.4f}".format(
                    epoch + 1, total_epoch, losses.avg))
             # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # version 1: don't update dstore even for finetuned version
            # version 2: update dstore each epoch
            self.model.eval()
            if not self.cfg.MODEL.FROZEN and self.cfg.MODEL.TYPE != "base":
                self.model.reset_cache()
                self.dstore_checkup(
                    train_loader, val_loader, epoch + 1
                )
                # update val and test knn features in self.model
                self._generate_train_features(val_loader)
                self._generate_train_features(test_loader)
                # print([*self.model.imageid2knnx.keys()])

            # Enable eval mode for dstore updating dstore and eval
            if self.cfg.NUM_GPUS == 1:
                self.model.eval()

                # eval at each epoch for single gpu training
                self.evaluator.update_iteration(epoch)
                # if not self.cfg.MODEL.FROZEN:
                #     self.eval_classifier(train_loader, "train")
                self.eval_classifier(val_loader, "val")
                if test_loader is not None:
                    self.eval_classifier(test_loader, "test")

                # check the patience
                t_name = "val_" + val_loader.dataset.name
                try:
                    curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
                except KeyError:
                    return

                if curr_acc > best_metric:
                    best_metric = curr_acc
                    best_epoch = epoch + 1
                    logger.info(
                        f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                    # if du.is_master_process(self.cfg.NUM_GPUS):
                    #     Checkpointer(
                    #         self.model,
                    #         save_dir=self.cfg.OUTPUT_DIR,
                    #         save_to_disk=True
                    #     ).save("best_model")
                    patience = 0
                else:
                    patience += 1
                if patience >= self.cfg.SOLVER.PATIENCE:
                    logger.info("No improvement. Breaking out of loop.")
                    break

        # save the last checkpoints as well
        if du.is_master_process(self.cfg.NUM_GPUS):
            Checkpointer(
                self.model,
                save_dir=self.cfg.OUTPUT_DIR,
                save_to_disk=True
            ).save("last_model")

    @torch.no_grad()
    def _generate_train_features(self, train_loader):
        train_base_features = []
        train_features, train_labels, train_ids = [], [], []
        for idx, input_data in enumerate(train_loader):
            # logger.info("getting dstore feature: {}/{}".format(
                # idx, len(train_loader)))
            X, targets, image_ids = self.get_input(input_data)
            X = X.to(self.device, non_blocking=True)
            base_x, knn_x = self.model(X, image_ids, return_feature=True)  # (batchsize, num_cls)

            # save the train features for updating dstore
            train_base_features.append(base_x.cpu().numpy())
            train_features.append(knn_x.cpu().numpy())
            # train_labels.append(targets.unsqueeze(1))  # batch_size x 1
            train_labels.extend(list(targets.numpy()))
            try:
                train_ids.extend(list(image_ids.numpy()))
            except AttributeError as e:
                # image_ids as list
                train_ids.extend(image_ids)
            del knn_x
            del base_x
            del X
            torch.cuda.empty_cache()
        # print("after cleaning: {:.2f} GB".format(gpu_mem_usage()))
        train_features = np.vstack(train_features)
        train_base_features = np.vstack(train_base_features)
        return train_features, train_labels, train_ids, train_base_features

    def _generate_train_features_dist(self, train_loader):
        dbank = dataBank()
        total_data = len(train_loader.dataset)
        for idx, input_data in enumerate(train_loader):
            if dbank.get_total() > total_data / 10:
                break
            X, targets, image_ids = self.get_input(input_data)
            X = X.to(self.device, non_blocking=True)

            _, knn_x = self.model(X, image_ids=None, return_feature=True)  # (batchsize, num_cls)

            # save the train features for updating dstore
            dbank.update(
                knn_x.cpu().numpy(),
                targets.numpy(),
                image_ids.numpy(),
            )

            del knn_x
            del X
            torch.cuda.empty_cache()
        # print("after cleaning: {:.2f} GB".format(gpu_mem_usage()))
        return dbank.get_feature()

    def update_dstore_saved(self, train_loader, epoch):
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

    def update_dstore(self, train_loader, epoch=None):
        logger.info("Updating dstore index")

        if self.cfg.NUM_GPUS > 1:
            t_features, t_labels, t_ids, _ = self._generate_train_features_dist(
                train_loader)
            # print(t_features.shape)
            # print(len(t_labels))
            # print(len(t_ids))
            self.model.module.dstore.update_index(
                t_features,
                np.array(t_labels)[:, np.newaxis],  # num_train x 1
                t_ids,
            )
        else:
            t_features, t_labels, t_ids, t_base_features = self._generate_train_features(train_loader)
            # compute nmi here
            # self.compute_nmi(t_features, t_labels, "knn")
            self.compute_nmi(t_base_features, t_labels, "base")
            self.model.dstore.update_index(
                t_features,
                np.array(t_labels)[:, np.newaxis],  # num_train x 1
                t_ids,
            )

        logger.info("...updated with train features size: {}".format(
            t_features.shape))

    def eval_classifier(self, data_loader, prefix):
        """evaluate classifier"""
        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name

        start_time = time.time()
        total = len(data_loader)
        num_warmup = min(5, log_interval - 1, total - 1)

        # initialize features and target
        total_joint_probs = []
        total_knn_probs = []
        total_base_probs = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
            X, targets, image_ids = self.get_input(input_data)
            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            with torch.no_grad():
                loss, outputs = self.forward_one_batch(
                    X, targets, image_ids, False)
            if loss == -1:
                return

            if (idx + 1) % log_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                if self.cfg.NUM_GPUS > 1:
                    k, log_temp, num_bins = self.model.module.dstore.get_params()
                else:
                    k, log_temp, num_bins = self.model.dstore.get_params()
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. ETA={} ".format(  # noqa
                        idx + 1,
                        total,
                        loss["main"],
                        seconds_per_img,
                        str(
                            datetime.timedelta(
                                seconds=int(seconds_per_img * (total - num_warmup) - duration)  # noqa
                            )
                        ),
                    ) + "log_temp={:.4f}, topk={:d}, num_bins={:d}".format(log_temp, k, num_bins)
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_joint_probs.append(outputs["probs"])
            if "base_probs" in outputs and outputs["base_probs"] is not None:
                total_knn_probs.append(outputs["knn_probs"])
                total_base_probs.append(outputs["base_probs"])

        out = {}
        # total_testimages x num_classes
        joint_probs = torch.cat(total_joint_probs, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_probs, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )
        out["targets"] = total_targets
        out["joint_probs"] = joint_probs

        if len(total_knn_probs) > 0:
            knn_probs = torch.cat(total_knn_probs, dim=0).cpu().numpy()
            base_probs = torch.cat(total_base_probs, dim=0).cpu().numpy()
            self.evaluator.classify(
                knn_probs, total_targets,
                test_name + "_knn", self.cfg.DATA.MULTILABEL,
            )
            self.evaluator.classify(
                base_probs, total_targets,
                test_name + "_base", self.cfg.DATA.MULTILABEL,
            )
            out["knn_probs"] = knn_probs
            out["base_probs"] = base_probs
        # save the probs and targets!
        out_path = os.path.join(
            self.cfg.OUTPUT_DIR, f"{test_name}_probs.pth")
        torch.save(out, out_path)
        logger.info(
            f"Saved probs and targets for {test_name} at {out_path}")


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (GB)."""
    if not torch.cuda.is_available():
        return 0
    # Number of bytes in a megabyte
    _B_IN_GB = 1024 * 1024 * 1024

    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / _B_IN_GB


def is_empty(feat_dict):
    empty = []
    total = len(feat_dict)
    for v in feat_dict.values():
        if isinstance(v, list) or isinstance(v, str):
            if len(v) == 0:
                empty.append(0)  # append false
        else:
            if torch.sum(torch.isnan(v)) > 0:
                empty.append(0)  # append false
    if len(empty) == total:
        return True
    return False


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class dataBank(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.train_features_list = []
        # self.train_base_features_list = []
        self.train_labels = []
        self.train_ids = []

    def get_feature(self):
        return np.vstack(self.train_features_list), self.train_labels, self.train_ids  #  np.vstack(self.train_base_features_list)

    def get_total(self):
        return len(self.train_features_list)

    def update(self, knn_x, targets, image_ids):
        self.train_features_list.append(knn_x)
        # self.train_base_features_list.append(base_x)
        self.train_labels.extend(list(targets))
        self.train_ids.extend(list(image_ids))
