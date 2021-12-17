#!/usr/bin/env python3
"""
Helper functions for knns
"""
import faiss
import math
import numpy as np
import torch


def euclidean_distance(x, y):
    """
    x -> torch tensor, size: batch_size x num_inputs1 x feature_dim, or num_inputs1 x feature_dim
    y -> torch tensor, size: batch_size x feature_dim x num_inputs2, or feature_dim x num_inputs2
    Returns:
        out: batch_size x num_inputs1 x num_inputs2, or num_inputs1 x num_inputs2
    """
    out = -2 * torch.einsum('ij,jk->ik', x, y)
    # out = -2 * torch.matmul(x, y)
    out += (x**2).sum(dim=-1, keepdim=True)
    out += (y**2).sum(dim=-2, keepdim=True)
    return out


def compute_cosine_sim(
    query_features, dstore_features, pool_type="neg", indices=None
):
    # try cosine (inner-product variant)
    sim = torch.einsum(
        'ij,jk->ik', query_features, dstore_features.permute(1, 0)
        )
    if indices is not None:
        sim = sim.gather(dim=-1, index=indices)
    return sim


def compute_sim(query_features, dstore_features, pool_type="neg", indices=None):
    """
    compute similarity (- pairwise distance) for all pairs of query items
    and all items in datastore.
    Args:
        query_features: num_queries x feature_dim
        dstore_features: num_dstore_items x feature_dim
        pool_type: str
        indices: if not None, num_quries x num_dstore_items'.
    Returns:
        sim: negative of the euclidean distance.
             shape: num_queries x num_dstore_items or
                    num_queries x num_dstore_items' if indices is not None
    """
    dist = euclidean_distance(query_features, dstore_features.permute(1, 0))
    # num_queries x num_dstore_items

    # D -> b m o
    if indices is not None:
        dist = dist.gather(dim=-1, index=indices)
    if pool_type.startswith("inv"):
        sims = 1 / dist
    elif pool_type.startswith("neg"):
        sims = -1 * dist
    return sims


def aggregate_output(W, x, indices, chunk_size=None):
    if chunk_size is not None:
        return aggregate_output_chuck(W, x, indices, chunk_size)
    return _aggregate_output(W, x, indices)


def aggregate_output_faiss(W, x, knn_indices):
    r"""
    Calculates weighted averages for k nearest neighbor features.
    Note that the results can be aggregated weights in feature space
    or in the output label space, depending on the choice x:
        - batch_size x hidden_dim if x is the train features
        - batch_Size x num_classes if x is the one-hot train labels
    Args:
        W: matrix of weights batch_size x k
        x: database items of shape num_train x feature_dims.
        knn_indices: index tensor of shape batch_size x k
    Returns:
        a batch_size x feature_dims tensor of the k nearest neighbor volumes for each query item
    """
    batch_size, k = W.shape
    num_train, feature_dim = x.shape
    x = x.expand(batch_size, -1, -1)  # batch_size x num_train x feature_dim
    knn_indices = knn_indices.view(batch_size, k, 1).expand(
        batch_size, k, feature_dim)

    # batch_size x k x feature_dim
    retrieval_transform = torch.gather(x, 1, knn_indices)

    knn_feature = torch.sum(
        torch.mul(
            retrieval_transform,
            W.view(batch_size, -1, 1),
        ),  # batch_size x k x feature_dim
        1,
    )  # batch_size x feature_dim
    return knn_feature


def _aggregate_output_pool(W, x, indices):
    r"""
    Calculates weighted averages for k nearest neighbor volumes.
    Note that the results can be aggregated weights in feature space
    or in the output label space, depending on the choice x:
        - batch_size x hidden_dim if x is the train features
        - batch_Size x num_classes if x is the one-hot train labels
    Args:
        W: matrix of weights batch_size x (num_train - 1) x k
        x: database items of shape num_train x feature_dims.
        indices: index tensor of shape batch_size x (num_train - 1)
    Returns:
        a batch_size x feature_dims x k tensor of the k nearest neighbor volumes for each query item
    """
    # print(W.shape, x.shape, indices.shape)
    b, o, k = W.shape
    n, e = x.shape

    # x_interm = x.view(1, n, e).detach()
    # x_interm = x.view(n, e).detach()

    # put W to the indexed position
    indices = indices.view(b, 1, o).expand(b, k, o)
    weights_full = torch.cuda.FloatTensor(b, k, n).fill_(0.)
    weights_full = weights_full.type(W.dtype)
    weights_full = weights_full.scatter_add_(
        src=W.permute(0, 2, 1), index=indices, dim=2)  # batch_size x k x num_train

    weights_full = torch.sum(weights_full, 1)  # batch_size x num_train
    return torch.einsum('ij,jk->ik', weights_full, x)


def _aggregate_output(W, x, indices):
    r"""
    Calculates weighted averages for k nearest neighbor volumes.
    Note that the results can be aggregated weights in feature space
    or in the output label space, depending on the choice x:
        - batch_size x hidden_dim if x is the train features
        - batch_Size x num_classes if x is the one-hot train labels
    Args:
        W: matrix of weights batch_size x (num_train - 1) x k
        x: database items of shape num_train x feature_dims.
        indices: index tensor of shape batch_size x (num_train - 1)
        train: whether to use tensor comprehensions for inference (forward only)
    Returns:
        a batch_size x feature_dims x k tensor of the k nearest neighbor volumes for each query item
    """
    # print(W.shape, x.shape, indices.shape)
    b, o, k = W.shape
    n, e = x.shape

    # x_interm = x.view(1, n, e).detach()
    x_interm = x.view(n, e).detach()

    # put W to the indexed position
    indices = indices.view(b, 1, o).expand(b, k, o)
    weights_full = torch.cuda.FloatTensor(b, k, n).fill_(0.)
    weights_full = weights_full.type(W.dtype)
    weights_full = weights_full.scatter_add_(
        src=W.permute(0, 2, 1), index=indices, dim=2)  # batch_size x k x num_train

    # TODO: make it add or max in this for loop,
    z_interm = torch.cuda.FloatTensor(b, e).fill_(0.)
    z_interm = z_interm.type(W.dtype)
    for i_k in range(k):
        # batch_size x 1 x emb_size
        z_interm += torch.einsum(
            'ij,jk->ik', weights_full[:, i_k, :], x_interm)
        # z_interm += torch.matmul(weights_full[:, i_k:i_k+1, :], x_interm).squeeze()
    del weights_full
    return z_interm
    # z_interm = torch.cat(
    #     [torch.matmul(
    #         weights_full[:, i_k:i_k+1, :], x_interm) for i_k in range(k)],
    #     1)  # batch_size x k x emb_size

    # del weights_full

    # return z_interm.permute(0, 2, 1)


def aggregate_output_chuck(W, x, indices, chunk_size):
    r"""
    Calculates weighted averages for k nearest neighbor volumes.
    Note that the results can be aggregated weights in feature space
    or in the output label space, depending on the choice x:
        - batch_size x hidden_dim if x is the train features
        - batch_Size x num_classes if x is the one-hot train labels
    Args:
        W: matrix of weights batch_size x (num_train - 1) x k
        x: database items of shape num_train x feature_dims.
        indices: index tensor of shape batch_size x (num_train - 1)
        chuck_size: to
    Returns:
        a batch_size x feature_dims x k tensor of the k nearest neighbor volumes for each query item
    """
    # print(W.shape, x.shape, indices.shape)
    b, o, k = W.shape
    n, e = x.shape

    x_interm = x.view(1, n, e).detach()
    z_chunks = []

    for b_offset in range(0, b, chunk_size):
        this_chunk_size = min(chunk_size, b - b_offset)
        I_chunk = indices[b_offset:b_offset+this_chunk_size, :]
        w_chunk = W[b_offset:b_offset+this_chunk_size, :, :]

        # put W to the indexed position
        indices_f = I_chunk.view(this_chunk_size,1,o).expand(
            this_chunk_size, k, o)
        weights_full = torch.cuda.FloatTensor(this_chunk_size, k, n).fill_(0.)
        weights_full = weights_full.type(w_chunk.dtype)
        weights_full = weights_full.scatter_add_(
            src=w_chunk.permute(0, 2, 1), index=indices_f, dim=2)  # chuck_size x k x num_train

        z_chunk = torch.cuda.FloatTensor(this_chunk_size, e).fill_(0.)
        z_chunk = z_chunk.type(W.dtype)
        for i_k in range(k):
            z_chunk += torch.matmul(weights_full[:, i_k:i_k+1, :], x_interm).squeeze()
            # batch_size x emb_size
        del weights_full
        del indices_f
        torch.cuda.empty_cache()
        z_chunks.append(z_chunk)
        # z_chunks.append(z_chunk.permute(0, 2, 1))

    return torch.cat(z_chunks, 0)


def log1mexp(x, expm1_guard = 1e-7):
    # See https://cran.r-project.org/package=Rmpfr/.../log1mexp-note.pdf
    t = x < math.log(0.5)
    y = torch.zeros_like(x)
    y[t] = torch.log1p(-x[t].exp())

    # for x close to 0 we need expm1 for numerically stable computation
    # we furtmermore modify the backward pass to avoid instable gradients,
    # ie situations where the incoming output gradient is close to 0 and the gradient of expm1 is very large
    # expxm1 = torch.expm1(x[1 - t])
    expxm1 = torch.expm1(x[t.logical_not()])

    log1mexp_fw = (-expxm1).log()
    log1mexp_bw = (-expxm1+expm1_guard).log()  # limits magnitude of gradient

    y[t.logical_not()] = log1mexp_fw.detach() + (log1mexp_bw - log1mexp_bw.detach())
    return y


def setup_index(features, use_l2=True):
    # build up the faiss index
    # step 1: build the index
    if use_l2:
        # l2
        index = faiss.IndexFlatL2(features.shape[1])
    else:
        # inner-product / cos-sim
        index = faiss.IndexFlatIP(features.shape[1])

    # step 2: add vectors to the index
    index.add(features)
    return index


class IndexedMatmul2Efficient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, indices, chunk_size=32):
        ctx.save_for_backward(x, W, indices)
        ctx.chunk_size = chunk_size

        b, o, k = W.shape
        n, e = x.shape

        x_interm = x.view(1, n, e).detach()
        z_chunks = []

        for b_offset in range(0, b, chunk_size):
            this_chunk_size = min(chunk_size, b - b_offset)
            I_chunk = indices[b_offset:b_offset+this_chunk_size, :]
            w_chunk = W[b_offset:b_offset+this_chunk_size, :, :]

            # put W to the indexed position
            indices_f = I_chunk.view(this_chunk_size,1,o).expand(
                this_chunk_size, k, o)
            weights_full = torch.cuda.FloatTensor(this_chunk_size, k, n).fill_(0.)
            weights_full = weights_full.type(w_chunk.dtype)
            weights_full = weights_full.scatter_add_(
                src=w_chunk.permute(0,2,1), index=indices_f, dim=2)  # batch_size x k x num_train

            z_chunk = torch.cat(
                [torch.matmul(
                    weights_full[:, i_k:i_k+1, :], x_interm) for i_k in range(k)],
                1)  # batch_size x k x emb_size
            z_chunks.append(z_chunk.permute(0, 2, 1))

        return torch.cat(z_chunks, 0)

    @staticmethod
    def backward(ctx, grad):
        x, W, indices = ctx.saved_tensors
        chunk_size = ctx.chunk_size

        b, o, k = W.shape
        n, e = x.shape

        x_interm = x.view(1, n, e).detach()
        grad_x = torch.zeros_like(x)
        grad_w_chunks = []

        for b_offset in range(0, b, chunk_size):
            this_chunk_size = min(chunk_size, b - b_offset)
            I_chunk = indices[b_offset:b_offset+this_chunk_size, :]
            w_chunk = W[b_offset:b_offset+this_chunk_size, :, :]

            grad_chunk = grad[b_offset:b_offset+this_chunk_size, :, :].permute(0,2,1)
            indices_f = I_chunk.view(this_chunk_size,1,o).expand(
                this_chunk_size, k, o)
            del I_chunk

            w_full = torch.cuda.FloatTensor(this_chunk_size, k, n).fill_(0)
            w_full = w_full.type(w_chunk.dtype)
            w_full = w_full.scatter_add(
                source=w_chunk.permute(0, 2, 1), index=indices_f, dim=2)
            del w_chunk

            for i_k in range(k):
                grad_x += torch.matmul(
                    grad_chunk[:, i_k, :], w_full[:, i_k, :]).permute(0, 2, 1)

            del w_full
            grad_w_full = torch.cat([
                torch.matmul(x_interm, grad_chunk[:,i_k:i_k+1,:]) \
                for i_k in range(k)
            ], 1)
            del grad_chunk
            grad_w_chunk = grad_w_full.gather(2, indices_f.permute(0,1,3,2)).permute(0,3,2,1)
            del grad_w_full
            grad_w_chunks.append(grad_w_chunk)

        grad_w = torch.cat(grad_w_chunks, 0)
        return grad_x, grad_w, None, None


def multi_scale(samples, model):
    # get multi-scaled knn features
    v = None
    for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = torch.nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
        feats = model(inp, image_ids=None, return_feature=True).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v


def entropy(probs, base):
    """probs: np.array, (n_classes,), sum to 1"""
    exponent = np.log(sanitycheck_probs(probs)) / np.log(base)
    return - np.multiply(exponent, probs).sum()


def sanitycheck_probs(probs):
    # check if there is any 0 values, if so, add a eps to that position
    probs = np.array(probs)
    return probs + (probs == 0) * 1e-16


def mean_entropy(y_probs):
    num_samples, num_classes = y_probs.shape
    entropy_list = []
    for i in range(num_samples):
        entropy_list.append(entropy(y_probs[i, :], num_classes))
    return "{:.6f} +- {:.6f}".format(np.mean(entropy_list), np.std(entropy_list))


