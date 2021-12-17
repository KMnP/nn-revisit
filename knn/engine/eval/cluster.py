#!/usr/bin/env python3
"""
evaluate NMI and triplet generalization errors
"""
import faiss
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, v_measure_score
from sklearn.metrics.pairwise import euclidean_distances
from typing import Any, List, Tuple


def build_knn_index(
        x: np.ndarray, nmb_clusters: int
) -> faiss.IndexFlatL2:
    """build faiss index and a Kmeans object"""
    dim = x.shape[1]
    # faiss implementation of k-means
    clus = faiss.Clustering(dim, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    index = faiss.IndexFlatL2(dim)
    # if faiss.get_num_gpus() > 0:
    #     index = faiss.index_cpu_to_all_gpus(index)
    # perform the training
    clus.train(x, index)
    return index


def run_kmeans(x: np.ndarray, nmb_clusters: int) -> List[int]:
    """
    modified from: https://github.com/facebookresearch/deepcluster
    assign labels to index in x
    Args:
        x: (num_test_samples, dim)
        nmb_clusters: how many cluster,
    Returns:
        pred_labels: list of predicted labels, len = num_test_samples
            labels here are in the range of [0, nmb_clusters)
            (num_samples,  k)
        distances: (num_samples, k)
    """
    x = np.ascontiguousarray(x.astype("float32"))
    index = build_knn_index(x, nmb_clusters)
    # compute the mapping from x to the cluster centroids
    # after kmeans has finished training
    distances, pred_labels = index.search(x, 1)

    return pred_labels, distances


def nmi(
        gt_labels: List[int], input_embeddings: np.ndarray, num_clusters: int
) -> float:
    """
    get NMI for clustering quality
    3 ways to play with this function:
    use different set of gt_labels, "train" "val" "test".
    Args:
        gt_labels: a list of gt labels,
            sequence is the same as input_embeddings in dim 0
        input_embeddings: (num_test_samples, dim)
        num_clusters
    Returns:
        nmi: float
    """
    assert len(gt_labels) == len(input_embeddings)
    pred_labels, _ = run_kmeans(input_embeddings, num_clusters)
    # print(pred_labels.shape)

    pred_labels = [int(n[0]) for n in pred_labels]
    gt_labels = [int(n) for n in gt_labels]
    nmi = normalized_mutual_info_score(gt_labels, pred_labels)
    a_nmi = adjusted_mutual_info_score(gt_labels, pred_labels)
    v_nmi = v_measure_score(gt_labels, pred_labels)

    return nmi, a_nmi, v_nmi
