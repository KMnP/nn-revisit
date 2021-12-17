#!/usr/bin/env python3
"""
a evaluator class using knn with faiss
"""
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import os
import pandas as pd

from collections import defaultdict
from numpy import linalg as LA
from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..engine.eval import cluster
from ..engine.eval.singlelabel import accuracy
from ..models.resnet import combine_knn_and_linear_probs
from ..utils import logging
from ..utils.io_utils import save_or_append_df
logger = logging.get_logger("nearest_neighbors")
FEAT_DIR=""  # TODO: move it to config


class KNNEvaluator():
    """
    a trainer with below logics:

    1. Build model, optimizer, scheduler, dataloader
    2. Load checkpoints if provided
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

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            checkpointer = Checkpointer(
                self.model,
                save_dir=cfg.OUTPUT_DIR,
                save_to_disk=True
            )
            checkpointer.load(cfg.MODEL.WEIGHT_PATH)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

            if self.cfg.DATA.FEATURE == "dino_resnet50":
                logger.info("re-naming some keys for dino_resnet50")
                # dino_resnet50 has some legacy issue: it was inside vit_backbone at first, and then moved to resnet later
                checkpoint = torch.load(
                    cfg.MODEL.WEIGHT_PATH, map_location="cpu")
                state_dict = checkpoint['model']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('enc.conv1'):
                        # remove prefix
                        state_dict["enc.0" + k[len("enc.conv1"):]] = state_dict[k]
                        del state_dict[k]
                    elif k.startswith('enc.bn1'):
                        # remove prefix
                        state_dict["enc.1" + k[len("enc.bn1"):]] = state_dict[k]
                        del state_dict[k]

                    elif k.startswith('enc.layer1'):
                        # remove prefix
                        state_dict["enc.4" + k[len("enc.layer1"):]] = state_dict[k]
                        del state_dict[k]

                    elif k.startswith('enc.layer2'):
                        # remove prefix
                        state_dict["enc.5" + k[len("enc.layer2"):]] = state_dict[k]
                        del state_dict[k]

                    elif k.startswith('enc.layer3'):
                        # remove prefix
                        state_dict["enc.6" + k[len("enc.layer3"):]] = state_dict[k]
                        del state_dict[k]

                    elif k.startswith('enc.layer4'):
                        # remove prefix
                        state_dict["enc.7" + k[len("enc.layer4"):]] = state_dict[k]
                        del state_dict[k]

                model.load_state_dict(state_dict, strict=False)

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")
        # this class does not train the model at all
        self.model.eval()
        if cfg.SOLVER.LOSS == "softmax" or cfg.SOLVER.LOSS == "knn_reg" or cfg.SOLVER.LOSS == "knn_reg_test" or cfg.SOLVER.LOSS == "knn_focalstar":
            self.activation = "softmax"
        else:
            self.activation = "sigmoid"

    def compute_nmi(self, train_features, train_targets):
        # see if need to normalize feature
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
            "Clutering nmi: {:.4f}, adjusted nmi: {:.4f}, v: {:.4f}".format(
                nmi, a_nmi, v_nmi
            ))

    def get_input(self, data):
        if not isinstance(data, dict):
            images, targets = data
            images = images.to(self.device)
            targets = targets.to(self.device)
            return images, targets, None

        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"]
        labels = data["label"]
        image_ids = data["id"]

        # move data to device
        inputs = inputs.to(self.device)        # (batchsize, 2048)
        labels = labels.to(self.device)        # (batchsize, )
        # image_ids = image_ids.to(self.device)  # (batchsize, )
        return inputs, labels, image_ids

    @torch.no_grad()
    def tune_knn_imagenet(self, val_features, val_labels, val_base_probs):
        # tune knn using smaller training set
        logger.info("Tuning temperature, k, coeff using imagenet val set")
        best_T, best_k, best_coeff = None, None, None
        best_acc = -1
        val_base_probs = torch.from_numpy(val_base_probs).type(torch.float64)
        base_acc, _ = accuracy(val_base_probs, val_labels)
        logger.info("base prob: {}".format(val_base_probs.shape))
        logger.info("val_features: {}".format(val_features.shape))

        save_name = os.path.join(self.cfg.OUTPUT_DIR, f"tune.pkl")
        out_prob_path = os.path.join(self.cfg.OUTPUT_DIR, f"val_probs.pth")
        for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            for T in [0.01, 0.1, 1, 10]:
                self.model.dstore.update_params(k, T)

                # do it in chunck to avoid out-of-memory issue
                X_test_list = np.array_split(val_features, 1000)
                knn_probs_list = []
                for X_test_chunk in X_test_list:
                    knn_probs_list.append(
                        self.model.dstore.get_knn_prob(X_test_chunk).cpu().numpy()
                    )
                probs = np.vstack(knn_probs_list)
                del knn_probs_list
                del X_test_list

                knn_acc, _ = accuracy(probs, val_labels)
                # tune for the joint probs for joint model
                # for coeff in [0.05, 0.001, 0.01, 0.1]
                probs = torch.from_numpy(probs).type(torch.float64)

                data_dict = defaultdict(list)
                for coeff in np.arange(0.05, 1, 0.05):  # 0.05 - 0.95
                    j_probs = combine_knn_and_linear_probs(
                        probs, val_base_probs, coeff, self.activation)
                    acc, _ = accuracy(j_probs.cpu().numpy(), val_labels)

                    data_dict["k"].append(k)
                    data_dict["T"].append(T)
                    data_dict["coeff"].append(coeff)
                    data_dict["knn_acc"].append(knn_acc * 100)
                    data_dict["joint_acc"].append(acc * 100)
                    data_dict["base_acc"].append(base_acc * 100)
                    # data_dict["knn_entropy"].append()
                    # data_dict["base_entropy"].append()

                    if acc > best_acc:
                        best_acc = acc
                        best_T, best_k = T, k
                        best_coeff = coeff
                        # save the j_probs
                        out = {
                            "joint_probs": j_probs,
                            "best": [best_T, best_k, best_coeff, best_acc]
                        }
                        torch.save(out, out_prob_path)

                df = pd.DataFrame(data_dict)
                save_or_append_df(save_name, df)

        for k in [best_k]:
            for T in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]:
                self.model.dstore.update_params(k, T)

                # do it in chunck to avoid out-of-memory issue
                X_test_list = np.array_split(val_features, 1000)
                knn_probs_list = []
                for X_test_chunk in X_test_list:
                    knn_probs_list.append(
                        self.model.dstore.get_knn_prob(X_test_chunk).cpu().numpy()
                    )
                probs = np.vstack(knn_probs_list)
                del knn_probs_list
                del X_test_list

                knn_acc, _ = accuracy(probs, val_labels)
                # tune for the joint probs for joint model
                probs = torch.from_numpy(probs).type(torch.float64)

                data_dict = defaultdict(list)
                for coeff in np.arange(0.05, 1, 0.05):  # 0.05 - 0.95
                    j_probs = combine_knn_and_linear_probs(
                        probs, val_base_probs, coeff, self.activation)
                    acc, _ = accuracy(j_probs.cpu().numpy(), val_labels)

                    data_dict["k"].append(k)
                    data_dict["T"].append(T)
                    data_dict["coeff"].append(coeff)
                    data_dict["knn_acc"].append(knn_acc * 100)
                    data_dict["joint_acc"].append(acc * 100)
                    data_dict["base_acc"].append(base_acc * 100)
                    # data_dict["knn_entropy"].append()
                    # data_dict["base_entropy"].append()

                    if acc > best_acc:
                        best_acc = acc
                        best_T, best_k = T, k
                        best_coeff = coeff
                        # save the probs
                        out = {
                            "joint_probs": j_probs,
                            "best": [best_T, best_k, best_coeff, best_acc]
                        }
                        torch.save(out, out_prob_path)

                df = pd.DataFrame(data_dict)
                save_or_append_df(save_name, df)

        # update:
        logger.info("Finish tuning dstore parameters: {}, {}, {}".format(
            best_k, best_T, best_coeff))

    @torch.no_grad()
    def tune_knn(self, val_loader):
        # tune knn using smaller training set
        logger.info("Tuning temperature, k, coeff using val set")
        total_base_probs = []
        total_val_features = []
        total_val_targets = []

        with torch.no_grad():
            for _, input_data in enumerate(val_loader):
                X, targets, image_ids = self.get_input(input_data)
                outputs = self.model(X, image_ids)  # (batchsize, num_cls)

                if "base_probs" in outputs:
                    total_base_probs.append(outputs["base_probs"])
                total_val_features.append(outputs["features"])
                total_val_targets.extend(list(targets.cpu().numpy()))

        if len(total_base_probs) > 0:
            total_base_probs = torch.cat(total_base_probs, dim=0)
        else:
            total_base_probs = None
        total_val_features = torch.cat(total_val_features, dim=0)

        # tune
        best_T, best_k, best_coeff = None, None, None
        best_acc, best_knn_acc = -1, -1

        save_name = os.path.join(self.cfg.OUTPUT_DIR, f"tune.pkl")
        data_dict = defaultdict(list)

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

                if self.cfg.DSTORE.TUNE_WITH_KNN_ONLY:
                    acc, _ = accuracy(probs.cpu().numpy(), total_val_targets)
                    if acc > best_acc:
                        best_acc = acc
                        best_T, best_k = T, k
                elif total_base_probs is not None:
                    # tune for the joint probs for joint model
                    for coeff in np.arange(0.05, 1, 0.05):  # 0.05 - 0.95
                        knn_acc, _ = accuracy(
                            probs.cpu().numpy(), total_val_targets)
                        j_probs = combine_knn_and_linear_probs(
                            probs, total_base_probs,
                            torch.tensor(coeff), self.activation)
                        acc, _ = accuracy(
                            j_probs.cpu().numpy(), total_val_targets)

                        # print(acc, T, k, coeff)
                        data_dict["k"].append(k)
                        data_dict["T"].append(T)
                        data_dict["coeff"].append(coeff)
                        data_dict["knn_acc"].append(knn_acc * 100)
                        data_dict["joint_acc"].append(acc * 100)

                        if acc > best_acc:
                            best_acc = acc
                            best_knn_acc = knn_acc
                            best_T, best_k = T, k
                            best_coeff = coeff
                        elif acc == best_acc and knn_acc > best_knn_acc:
                            best_acc = acc
                            best_knn_acc = knn_acc
                            best_T, best_k = T, k
                            best_coeff = coeff

        df = pd.DataFrame(data_dict)
        save_or_append_df(save_name, df)
        # update:
        logger.info("Finish tuning dstore parameters: {}, {}, {}".format(
            best_k, best_T, best_coeff))
        self.model.dstore.update_params(best_k, best_T)
        if best_coeff is not None:
            self.model.coeff = torch.tensor(best_coeff)

    def _generate_train_features(self, train_loader):
        with torch.no_grad():
            train_features, train_labels, train_ids = [], [], []
            for idx, input_data in enumerate(train_loader):
                X, targets, image_ids = self.get_input(input_data)

                _, knn_x = self.model(X, image_ids=None, return_feature=True)  # (batchsize, num_cls)

                # save the train features for updating dstore
                train_features.append(knn_x.cpu().numpy())
                # train_labels.append(targets.unsqueeze(1))  # batch_size x 1
                train_labels.extend(list(targets.cpu().numpy()))
                train_ids.extend(list(image_ids.numpy()))
                del knn_x
                del X
                torch.cuda.empty_cache()
            train_features = np.vstack(train_features)
        return train_features, train_labels, train_ids

    @torch.no_grad()
    def build_dstore_imagenet(self, train_loader, val_loader):
        logger.info("Building dstore index")
        num_imgs, num_clses = 1281167, 1000
        num_val_imgs = 50000
        m2featdim = {
            "sup_vitb16_224": 768,
            "sup_vitb16": 768,
            "sup_vitl16_224": 1024,
            "sup_vitl16": 1024,
            "sup_vitb16_imagenet22k": 768,
            "sup_vitl16_imagenet22k": 1024,
            "mocov3_vits": 384, "mocov3_vitb": 768,
            "swint_imagenet": 768,
            "swins_imagenet": 768,
            "swinb_imagenet_224": 1024,
            "swint_imagenet_ssl": 768,
            "sup_xcit_small_12_p16": 384,
            "sup_xcit_small_12_p8": 384,
            "sup_xcit_medium_24_p16": 512,
            "sup_xcit_medium_24_p8": 512,

            "dino_xcit_small_12_p16": 384,
            "dino_xcit_medium_24_p16": 512,
            "dino_xcit_small_12_p8": 384,
            "dino_xcit_medium_24_p8": 512,
            # "dino_resnet50"
        }
        feat_dim = m2featdim.get(self.cfg.DATA.FEATURE, 2048)
        logger.info(f"feature dim = {feat_dim} for {self.cfg.DATA.FEATURE}")

        # check if there is a saved features, works only for linear evals
        r = FEAT_DIR
        prefix = os.path.join(r, "{}_{}_{}".format(
            self.cfg.DATA.FEATURE, self.cfg.DATA.NAME,
            self.cfg.DSTORE.FEATUREMAP_POOL))

        if os.path.exists(prefix + '_trainfeatures.npy'):
            # only loading previously computed features for linear evaluation
            logger.info(
                f"loading previously saved train feature from {prefix}")

            train_features = np.memmap(
                prefix + '_trainfeatures.npy',
                dtype=np.float32, mode='r', shape=(num_imgs, feat_dim)
            )
            train_labels = np.memmap(
                prefix + '_trainlabels.npy', dtype=np.int, mode='r',
                shape=(num_imgs, 1)
            )
            image_ids = np.memmap(
                prefix + '_trainids.npy', dtype=np.int, mode='r',
                shape=(num_imgs, 1)
            )
        else:
            train_features = np.memmap(
                prefix + '_trainfeatures.npy', dtype=np.float32, mode='w+',
                shape=(num_imgs, feat_dim))
            train_labels = np.memmap(
                prefix + '_trainlabels.npy', dtype=np.int, mode='w+',
                shape=(num_imgs, 1)
            )
            image_ids = np.memmap(
                prefix + '_trainids.npy', dtype=np.int, mode='w+',
                shape=(num_imgs, 1)
            )
            t_features, t_labels, img_ids = self._generate_train_features(
                train_loader)
            t_features = t_features.astype(np.float32)
            t_labels = np.array(t_labels)[:, np.newaxis].astype(np.int)
            img_ids = np.array(img_ids)[:, np.newaxis].astype(np.int)

            train_features[:, :] = t_features[:, :]
            train_labels[:, :] = t_labels[:, :]
            image_ids[:, :] = img_ids[:, :]
            train_features.flush()
            train_labels.flush()
            image_ids.flush()

        self.model.dstore.update_index(
            train_features,
            train_labels,
            image_ids,
        )
        logger.info("...updated with train features size: {}".format(
            train_features.shape))

        if os.path.exists(prefix + '_valfeatures.npy'):
            logger.info(
                f"loading previously saved val feature from {prefix}")

            val_features = np.memmap(
                prefix + '_valfeatures.npy',
                dtype=np.float32, mode='r', shape=(num_val_imgs, feat_dim)
            )
            val_labels = np.memmap(
                prefix + '_vallabels.npy', dtype=np.int, mode='r',
                shape=(num_val_imgs, 1)
            )
            val_base_probs = np.memmap(
                prefix + '_valbaseprobs.npy',
                dtype=np.float32, mode='r', shape=(num_val_imgs, num_clses)
            )
        else:
            logger.info("getting validation features / probs / targets...")

            total_base_probs = []
            total_val_features = []
            total_val_targets = []

            with torch.no_grad():
                for _, input_data in enumerate(val_loader):
                    X, targets, _ = self.get_input(input_data)
                    outputs = self.model(X)  # (batchsize, num_cls)

                    total_base_probs.append(outputs["base_probs"].cpu().numpy())
                    total_val_features.append(outputs["features"].cpu().numpy())
                    total_val_targets.extend(list(targets.cpu().numpy()))
                    del outputs
                    del X
                    torch.cuda.empty_cache()

            total_base_probs = np.vstack(total_base_probs)
            total_val_features = np.vstack(total_val_features)
            total_val_features = total_val_features.astype(np.float32)
            total_base_probs = total_base_probs.astype(np.float32)
            total_val_targets = np.array(total_val_targets)[:, np.newaxis].astype(np.int)

            val_features = np.memmap(
                prefix + '_valfeatures.npy', dtype=np.float32, mode='w+',
                shape=(num_val_imgs, feat_dim))
            val_labels = np.memmap(
                prefix + '_vallabels.npy', dtype=np.int, mode='w+',
                shape=(num_val_imgs, 1)
            )
            val_base_probs = np.memmap(
                prefix + '_valbaseprobs.npy',
                dtype=np.float32, mode='w+', shape=(num_val_imgs, num_clses)
            )
            val_features[:, :] = total_val_features[:, :]
            val_base_probs[:, :] = total_base_probs[:, :]
            val_labels[:, :] = total_val_targets[:, :]
            val_features.flush()
            val_base_probs.flush()
            val_labels.flush()

        try:
            self.compute_nmi(train_features, train_labels)
        except Exception as e:
            print(e)
        return val_features, val_labels, val_base_probs

    def build_dstore(self, train_loader):
        logger.info("Building dstore index")

        # check if there is a saved features, works only for linear evals
        if self.cfg.DATA.CROPSIZE == 224:
            postfix = ""
        else:
            postfix = "_448"
        outfile = os.path.join(FEAT_DIR, "{}_{}_{}{}.pth".format(
            self.cfg.DATA.FEATURE, self.cfg.DATA.NAME,
            self.cfg.DSTORE.FEATUREMAP_POOL, postfix
        ))
        if os.path.exists(outfile) and self.cfg.MODEL.FROZEN:
            # only loading previously computed features for linear evaluation
            logger.info(
                f"loading previously saved train feature from {outfile}")
            data = torch.load(outfile, map_location=torch.device('cpu'))
            train_features = data["train_features"]
            train_labels = data["train_labels"]  # (num_train, 1)
            image_ids = data["image_ids"]  # (num_train, 1)
        else:
            # # in practice, will not use this block
            # raise ValueError("should have saved the features from previous training for linear eval")
            logger.info("using updated backbone features")
            train_features, t_labels, image_ids = self._generate_train_features(
                train_loader)
            train_labels = np.array(t_labels)[:, np.newaxis]

        self.model.dstore.update_index(
            train_features,
            train_labels,
            image_ids,
        )
        logger.info("...updated with train features size: {}".format(
            train_features.shape))
        self.compute_nmi(train_features, train_labels)

    def get_results(self, train_loader, val_loader, test_loader, build=True):
        if self.cfg.DATA.NAME == "imagenet":
            raise ValueError("this method only used for none-imagenet data")
        # tune dstore
        num_imgs, num_clses = train_loader.dataset.get_info()
        self.meank = int(num_imgs / num_clses)
        self.maxk = num_clses

        # tune knn using validation
        if build:
            self.build_dstore(train_loader)
        self.tune_knn(val_loader)

        # evaluate both val and test
        self.evaluator.update_iteration(0)
        self.eval_classifier(val_loader, "val", use_cache=not build)
        if test_loader is not None:
            self.eval_classifier(test_loader, "test", use_cache=not build)

    def eval_classifier(self, data_loader, prefix, use_cache=True):
        """evaluate classifier"""
        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        if use_cache:
            test_name = prefix
        else:
            test_name = f"finetuned_{prefix}"

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
                if use_cache:
                    outputs = self.model(X, image_ids)  # (batchsize, num_cls)
                else:
                    outputs = self.model(X)

            if (idx + 1) % log_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                logger.info(
                    "\tTest {}/{}. {:.4f} s / batch. ETA={} ".format(  # noqa
                        idx + 1,
                        total,
                        seconds_per_img,
                        str(
                            datetime.timedelta(
                                seconds=int(seconds_per_img * (total - num_warmup) - duration)  # noqa
                            )
                        ),
                    )
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
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
        out_path = os.path.join(self.cfg.OUTPUT_DIR, f"{test_name}_probs.pth")
        torch.save(out, out_path)
        logger.info(f"Saved probs and targets for {test_name} at {out_path}")
