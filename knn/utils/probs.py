import numpy as np


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

