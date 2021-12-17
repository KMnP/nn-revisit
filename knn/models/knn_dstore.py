#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

from numpy import linalg as LA

from .knn_utils import (
    aggregate_output, aggregate_output_faiss, compute_sim, compute_cosine_sim, log1mexp, setup_index
)
from ..utils.io_utils import get_feature_df
from ..utils import logging
logger = logging.get_logger("nearest_neighbors")
warnings.filterwarnings("ignore")  # ignore the multiple overflow warning


class KNNBase(nn.Module):
    def __init__(self, cfg, cur_device):
        super(KNNBase, self).__init__()
        # for knn-classifier
        self.k = cfg.DSTORE.TOPK
        self.T = cfg.DSTORE.TEMP
        self.norm_type = cfg.DSTORE.NORM_TYPE
        self.pool_type = cfg.DSTORE.POOL_TYPE
        self.dist_type = cfg.DSTORE.DIST_TYPE

        if not self.pool_type.endswith("sum"):
            raise ValueError("only support sum now for dstore")

        self.index = None
        self.device = cur_device
        self.num_classes = cfg.DATA.NUMBER_CLASSES

        # self.setup_faiss(cfg)

    def update_k(self, k):
        self.k = k

    def update_params(self, k, T):
        self.k = k
        self.T = T

    def _normalize_features(self, features):
        # normalize
        if self.norm_type is not None and self.norm_type.startswith("L2"):
            if self.norm_type == "L2NC":
                # norm_types: L2NC, center + l2 norm
                features = features - self.train_mean
            features = F.normalize(features, p=2, dim=1).to(self.device)
            #     train_features = train_features - self.train_mean
            # train_features = train_features / LA.norm(
            #     train_features, 2, 1)[:, None]
        return features

    def _prepare_features(self, train_features, train_labels):
        self.train_features, self.train_labels = None, None

        if isinstance(train_labels, torch.Tensor):
            self.train_labels = train_labels  # train_num x 1
        else:
            self.train_labels = torch.from_numpy(train_labels.astype(np.int))
            self.train_labels = self.train_labels.to(self.device)

        if not isinstance(train_features, torch.Tensor):
            train_features = torch.from_numpy(train_features.astype(np.float32))
            train_features = train_features.to(self.device)
        else:
            train_features = train_features.float()

        self.train_mean = train_features.mean(0).to(self.device)  # (2048, )

        # get number of classes:
        self.feature_dim = train_features.shape[1]

        self.train_features = self._normalize_features(train_features)
        #TODO: convert to fp16 if needed for larger training dataset

    def _setup_index(self):
        raise NotImplementedError()

    def setup_faiss(self, cfg):
        # setup index using the pretrained feature for the first epoch, if there is no saved features, leave it alone
        # Load in the features extracted by this model
        feature_df = get_feature_df(cfg)
        dataset_name = cfg.DATA.NAME
        _df = feature_df[feature_df["name"] == dataset_name].iloc[0]
        train_labels = _df['y_train'][:, np.newaxis]
        train_features = _df["X_train"]
        self._prepare_features(train_features, train_labels)

        # build the index
        self._setup_index()

    def update_index(self, train_features, train_labels, train_image_ids=None):
        if train_image_ids is not None:
            self.imageid2idx = {
                int(img_id): idx for idx, img_id in enumerate(list(train_image_ids))
            }
        # update train_features and train_labels
        self._prepare_features(train_features, train_labels)
        # build the index
        self._setup_index()

    def get_knns(self, queries):
        raise NotImplementedError()


class KNNtorch(KNNBase):
    """
    non-differentiable knn, the temp is also a variable, tuned in the process
    """
    def __init__(self, cfg, cur_device):
        super(KNNtorch, self).__init__(cfg, cur_device)
        self.return_probs = cfg.DSTORE.RETURN_PROBS
        self.use_cache = False  #cfg.DSTORE.USE_CACHE
        self.trainid2cache = {}
        self.knn_percentage = cfg.DATA.KNN_PERCENTAGE

    def get_params(self):
        return self.k, self.T, self.num_bins

    def _prepare_features(self, train_features, train_labels):
        self.train_features, self.train_labels = None, None

        if isinstance(train_labels, torch.Tensor):
            self.train_labels = train_labels.to(self.device)  # train_num x 1
        else:
            self.train_labels = torch.from_numpy(train_labels.astype(np.int))
            self.train_labels = self.train_labels.to(self.device)

        if not isinstance(train_features, torch.Tensor):
            train_features = torch.from_numpy(train_features.astype(np.float32))
            train_features = train_features.to(self.device)
        else:
            train_features = train_features.float().to(self.device)

        self.train_mean = train_features.mean(0).to(self.device)  # (2048, )

        # get number of classes:
        self.feature_dim = train_features.shape[1]

        self.train_features = nn.Parameter(
            self._normalize_features(train_features), requires_grad=False)

    def _setup_index(self):
        # there is no need for the index object here
        self.index = None

    def get_indices(self, queries, q_ids):
        # get the indices, so the dstore does not overlap with queries
        # return a tensor of shape batch_size x (num_train - 1)
        if self.knn_percentage < 1.0:
            # in this setting, all data from knn data store does not contain training data, so do not need to filter indices
            q_ids = None
        q_num = queries.shape[0]
        d_num = self.train_features.shape[0]
        if q_ids is not None:
            for q_id in q_ids:
                assert int(q_id) in self.imageid2idx, f"{q_id} not in the data store"

            # batch_size x num_train
            all_indices = torch.range(
                0, d_num - 1, dtype=torch.int64, device=self.device
            ).repeat(q_num, 1)
            ids = torch.tensor(
                [self.imageid2idx[int(idx)] for idx in q_ids],
                device=self.device
            )
            mask = torch.ones_like(
                all_indices).scatter_(1, ids.unsqueeze(1), 0.)
            # batch_size x (num_train - 1)
            indices = all_indices[mask.bool()].view(q_num, d_num - 1)
        else:
            indices = torch.range(
                0, d_num - 1, dtype=torch.int64, device=self.device
            ).repeat(q_num, 1)

        return indices

    def get_knns(self, queries, indices):
        # queries: torch.tensor of shape (n, dim)
        # step 1: preprocess accroding to the norm type
        # batch_size x feat_dim
        queries = self._normalize_features(queries)

        # step 2: compute the pariwise distance, self.train_features, self.train_labels
        # (feat_fim x batch_size) x (num_train x feat_dim)
        # --> batch_size x num_train or (num_train - 1)
        if self.dist_type == "l2":
            all_sims = compute_sim(
                queries, self.train_features, self.pool_type, indices)
        else:
            all_sims = compute_cosine_sim(
                queries, self.train_features, self.pool_type, indices)
        return all_sims

    def one_hot(self, labels):
        # return one-hot encoding for the provided labels
        y_onehot = torch.FloatTensor(labels.shape[0], self.num_classes)

        y_onehot.zero_()
        y_onehot = y_onehot.to(self.device)
        y_onehot.scatter_(1, labels, 1)
        return y_onehot

    @torch.no_grad()
    def forward(self, queries, q_ids=None):
        # a wrapper fn for forward
        if self.return_hist:
            return self.get_dist_histogram(queries, q_ids)
        return self.get_knn_prob(queries, q_ids)

    @torch.no_grad()
    def get_knn_prob(self, queries, q_ids=None):
        """
        return knn logits (unnormalized)
        """
        # check if need to compute out or use cache
        if q_ids is not None and self.use_cache:
            # assume all / none ids are in cache
            total = 0
            for q_id in q_ids:
                total += int(int(q_id) in self.trainid2cache)
            # assert total == 0 or total == len(q_ids)
            if total == len(q_ids):
            # if int(q_id) in self.trainid2cache:
                # print(f"total images in cache: {len(self.trainid2cache)}")
                out = []
                for q_id in q_ids:
                    out.append(self.trainid2cache[int(q_id)])
                return torch.cat(out, 0)

        # queries: numpy.ndarray or torch.tensor of shape (n, dim)
        # using cpus and queries has to be numpy array
        if not isinstance(queries, torch.Tensor):
            queries = torch.from_numpy(queries)
            queries = queries.to(self.device)

        indices = self.get_indices(queries, q_ids)

        # step 1: obtain the pair-wise sims: batch_size x num_train - 1
        sims = self.get_knns(queries, indices)
        sims, knn_indices = torch.topk(
            sims, self.k, largest=True, sorted=True)  # batch_size x k

        # step 2: # compute aggregation weights
        W = sims.clone().div_(self.T).exp_()  # batch_size x k


        # step 3: aggregate output
        if self.return_probs:
            # transform to one-hot
            x = self.one_hot(self.train_labels)  # num_train x num_classes
        else:
            x = self.train_features              # num_train x feat_dim
        # batch_num x num_classes / feat_dim
        out = aggregate_output_faiss(W, x, knn_indices)

        # save it to cache if needed
        if q_ids is not None and self.use_cache:
            for i in range(len(q_ids)):
                # (1, num_classes / feat_dim)
                self.trainid2cache[int(q_ids[i])] = out[i, :].unsqueeze(0)
        return out

