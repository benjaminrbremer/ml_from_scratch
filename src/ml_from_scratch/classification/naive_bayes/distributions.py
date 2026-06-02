"""
Statistical distribution helpers for Gaussian Naive Bayes.

The single exported function computes the per-class log-likelihood of every
sample under independent Gaussian distributions — one Gaussian per feature per
class. This is the "likelihood" half of Bayes' theorem; the class adds the
prior and does the argmax.

Why log-space?
--------------
The per-sample joint likelihood is the *product* of K*n individual Gaussian
PDFs (K classes, n features). For even modest n, this product underflows to
zero in floating-point. Working in log-space converts the product to a sum and
keeps the computation numerically stable throughout.

Why a separate module?
----------------------
Same reason loss.py is separate from logistic_regression.py: the distribution
math is pure and stateless — given arrays, return an array. The class owns the
learned parameters and the prediction logic; this module owns the math.
"""

import numpy as np


def gaussian_log_likelihood(
    X: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """
    Log-likelihood log P(x | y = k) for every sample and every class.

    Under the Gaussian Naive Bayes model, each feature x_j is drawn
    independently from a Gaussian conditioned on the class:

        x_j | y = k  ~  N(mu_kj, sigma_kj^2)

    The naive (independence) assumption lets us write the joint likelihood
    as a product over features:

        P(x | y = k) = prod_j  N(x_j ; mu_kj, sigma_kj^2)

    In log-space this becomes a sum:

        log P(x | y = k) = sum_j  log N(x_j ; mu_kj, sigma_kj^2)
                         = sum_j  -0.5 * [log(2 pi sigma_kj^2)
                                           + (x_j - mu_kj)^2 / sigma_kj^2]

    The computation uses NumPy broadcasting to evaluate all N samples against
    all K classes simultaneously -- no Python loops over samples or classes.

        X[:, np.newaxis, :]     has shape (N, 1, n)
        means[np.newaxis, :, :] has shape (1, K, n)
        diff                    has shape (N, K, n)
        log_ll                  has shape (N, K, n)
        return log_ll.sum(2)    has shape (N, K)

    Parameters
    ----------
    X : np.ndarray, shape (N, n)
        Input feature matrix -- N samples, n features.
    means : np.ndarray, shape (K, n)
        Per-class, per-feature means mu_kj.
    variances : np.ndarray, shape (K, n)
        Per-class, per-feature variances sigma_kj^2. Must be strictly positive
        (caller is responsible for variance smoothing before calling this).

    Returns
    -------
    np.ndarray, shape (N, K)
        log P(x_i | y = k) for every sample i and class k. The (i, k) entry
        is the sum over features of the log Gaussian PDF evaluated at x_i
        given the class-k parameters.
    """
    diff = X[:, np.newaxis, :] - means[np.newaxis, :, :]  # (N, K, n)
    log_ll = -0.5 * (np.log(2.0 * np.pi * variances) + diff**2 / variances)
    return log_ll.sum(axis=2)  # (N, K)
