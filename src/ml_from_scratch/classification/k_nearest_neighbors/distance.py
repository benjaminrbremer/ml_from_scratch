"""
Distance metrics for k-nearest neighbors.

Each function takes two matrices of points and returns the full pairwise
distance matrix between them:

    X_query : (M, n) matrix of query points  -- M samples, n features
    X_train : (N, n) matrix of stored points -- N samples, n features

and returns:

    D : (M, N) matrix where D[i, j] = distance(X_query[i], X_train[j]).

The KNN class calls one of these functions once per `predict` to build the
M x N distance matrix, then takes the k smallest entries per row.

Why a separate module?
----------------------
Same reason loss.py is separate from logistic_regression.py: the distance
math is pure and stateless -- given arrays, return an array. The class
owns the stored training set and the voting logic; this module owns the
metric.

Why only Euclidean here?
------------------------
This file implements Euclidean (L^2) distance only. It is by far the most
common choice and is sufficient for the figures and worked example in the
companion markdown. Several alternatives -- Manhattan (L^1), Chebyshev
(L^infinity), Minkowski (L^p), cosine, and distances for non-numeric data
(Hamming, Levenshtein, Jaccard, Gower) -- are discussed in the markdown's
Extensions section. Each would slot in as another function here, and
`DISTANCE_METRICS` plus the class dispatch would gain one more entry.

Conventions used throughout
---------------------------
    * All functions are fully vectorized: no Python loop over samples.
    * Inputs are 2D matrices. The class normalizes 1D inputs to 2D before
      calling these functions, so we can keep the signatures simple here.
    * The return is always shape (M, N) -- query points along rows, stored
      training points along columns. The class picks the per-row argmin
      neighbors with `np.argpartition`.
"""

import numpy as np

DISTANCE_METRICS = ["Euclidean"]


def euclidean(X_query: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """
    Pairwise Euclidean (L^2) distance between two sets of points.

    Definition
    ----------
    For two vectors a, b in R^n,

        d(a, b) = sqrt( sum_{j=1}^{n} (a_j - b_j)^2 ).

    Vectorized identity
    -------------------
    The direct definition would build an (M, N, n) tensor of pairwise
    differences and reduce along the last axis -- O(M * N * n) memory.
    We avoid that by expanding the square:

        || a - b ||^2 = || a ||^2 + || b ||^2 - 2 a . b.

    Applied to all M*N pairs at once:

        sq_query[i]      = sum_j X_query[i, j]^2     -- shape (M,)
        sq_train[j]      = sum_j X_train[j, j']^2    -- shape (N,)
        dot[i, j]        = X_query[i] . X_train[j]   -- shape (M, N)
        D2[i, j]         = sq_query[i] + sq_train[j] - 2 * dot[i, j]
        D[i, j]          = sqrt(D2[i, j])

    The whole thing is two outer-product broadcasts plus one matmul --
    O(M * N) memory (excluding the matmul's internal cost), and BLAS does
    the heavy lifting in the dot product.

    Numerical note
    --------------
    Subtracting two nearly-equal large numbers can produce tiny negative
    values for the squared distance due to floating-point roundoff (the
    true value is >= 0 but the computed one may be e.g. -1e-15). We clip
    to 0 before the sqrt to avoid NaN.

    Parameters
    ----------
    X_query : np.ndarray, shape (M, n)
        Query points -- one row per point at which we want neighbors.
    X_train : np.ndarray, shape (N, n)
        Stored training points -- one row per point we may treat as a
        neighbor.

    Returns
    -------
    np.ndarray, shape (M, N)
        D[i, j] = Euclidean distance between X_query[i] and X_train[j].
    """
    sq_query = np.sum(X_query**2, axis=1)  # (M,)
    sq_train = np.sum(X_train**2, axis=1)  # (N,)
    # Broadcasting: sq_query[:, None] is (M, 1); sq_train[None, :] is (1, N);
    # their sum is (M, N). Subtract 2 * X_query @ X_train.T (also (M, N)).
    sq_dists = sq_query[:, None] + sq_train[None, :] - 2.0 * (X_query @ X_train.T)
    # Clip away tiny negative values introduced by roundoff before sqrt.
    return np.sqrt(np.maximum(sq_dists, 0.0))
