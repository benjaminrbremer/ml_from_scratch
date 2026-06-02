"""
Binary k-nearest neighbors classifier -- brute-force, instance-based.

Model
-----
There is no parameter vector and no training loop. The "model" is just the
stored training set, and prediction is:

    for each query x:
        1. compute the distance from x to every stored training point
        2. pick the k smallest -- the k nearest neighbors
        3. return the class-1 probability as the proportion of those k
           neighbors with label 1, then optionally threshold for a class

This makes KNN a *non-parametric*, *instance-based*, *lazy* learner. The
contrast with logistic regression is the main pedagogical point:

    logistic_regression           k_nearest_neighbors
    ------------------------      -----------------------------
    parametric (w, b)             non-parametric (stores X_train)
    iterative gradient descent    O(1) "fit" -- just memorize
    O(n) prediction               O(N * n) prediction (brute force)
    linear decision boundary      piecewise boundary made of arcs
                                  between training points

`predict` here mirrors `logistic_regression.predict` in returning a
*probability*, not a label. `predict_class` thresholds it the same way
`logistic_regression.predict_class` does. The companion markdown derives
why we treat the k-neighbor proportion as a probability estimate.
"""

import numpy as np

import ml_from_scratch.classification.k_nearest_neighbors.distance as distance


class k_nearest_neighbors:
    """
    A binary k-nearest neighbors classifier using Euclidean distance and
    uniform (unweighted) majority voting.

    Attributes
    ----------
    k : int
        Number of neighbors to consult per prediction.
    n_features : int
        Number of input features the model expects. Set by `fit`.
    X_train : np.ndarray, shape (N, n_features)
        Stored training inputs. Set by `fit`.
    y_train : np.ndarray, shape (N,), dtype int
        Stored training labels, each in {0, 1}. Set by `fit`.
    """

    k: int
    n_features: int
    X_train: np.ndarray
    y_train: np.ndarray

    def __init__(self, k: int = 5):
        """
        Parameters
        ----------
        k : int, default 5
            Number of neighbors. Must be >= 1. For binary classification
            an odd k avoids ties in the vote, but tied votes are
            otherwise broken by returning the class with the smaller
            label (0) -- the np.round below-default rule.

        Notes
        -----
        Unlike logistic_regression, there is no `n_features` argument here:
        we cannot fix the feature count at construction time because we
        have not seen any data yet. It is set during `fit` from the shape
        of the training matrix.
        """
        if k < 1:
            raise ValueError("k must be >= 1")

        self.k = int(k)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        "Train" the model by storing the training set.

        There is no iteration and no parameter update. We validate shapes
        and label values, copy X and y onto the instance, and return. The
        whole computation cost of KNN lives in `predict`, not here.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)
            Training inputs.
        y : np.ndarray, shape (N,)
            Training labels, each in {0, 1}.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N, n_features); got shape {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(
                f"y must have shape (N,) matching X.shape[0]={X.shape[0]}; "
                f"got {y.shape}"
            )
        unique = np.unique(y)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError(f"y must contain only 0 or 1; got unique values {unique}")
        if X.shape[0] < self.k:
            raise ValueError(
                f"Cannot fit with N={X.shape[0]} samples and k={self.k}; "
                f"need at least k samples"
            )

        self.n_features = X.shape[1]
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the model to a batch of inputs and return class-1
        probabilities.

        For each query point x_i:

            1. d_i = euclidean(x_i, X_train)            -- shape (N,)
            2. take the k indices with the smallest d_i -- the k neighbors
            3. p_i = (1 / k) * sum_{j in neighbors} y_train[j]

        Stacked over M queries this is a single (M, N) distance matrix,
        a single argpartition along axis 1 to pick the k smallest per
        row, and a single mean of the gathered labels. No Python loop
        over query points.

        Why a proportion is a probability estimate
        ------------------------------------------
        Treat the k labels in the neighborhood of x as a sample of size k
        drawn from P(y = 1 | x). The sample mean is the maximum-likelihood
        estimate of P(y = 1 | x) under that model. For small k it is a
        noisy estimate (high variance); for large k it averages over a
        bigger neighborhood and gets smoother but more biased -- which is
        exactly the bias-variance trade-off in k discussed in the
        companion markdown.

        Parameters
        ----------
        X : np.ndarray
            Either shape (M, n_features) for a batch of M queries, or
            shape (n_features,) for a single query (reshaped internally).

        Returns
        -------
        np.ndarray, shape (M,)
            Predicted probabilities in [0, 1] -- specifically the
            fraction of the k nearest neighbors with label 1. Granular:
            each value is one of 0/k, 1/k, ..., k/k.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (M, {self.n_features}); got {X.shape}")

        # (M, N) pairwise distances. The class is committed to Euclidean.
        dists = distance.euclidean(X, self.X_train)

        # For each query row, take the indices of the k smallest distances.
        # argpartition is O(N) per row -- we don't need them sorted, just
        # to know which k they are. partition kth uses 0-based indexing.
        neighbor_idx = np.argpartition(dists, kth=self.k - 1, axis=1)[:, : self.k]

        # Gather the labels of those neighbors: shape (M, k).
        neighbor_labels = self.y_train[neighbor_idx]
        # Mean along axis 1 is the class-1 proportion for each query.
        return neighbor_labels.mean(axis=1)

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Apply the model and return discrete class labels {0, 1} by
        thresholding the predicted probabilities.

        For odd k and the default threshold of 0.5 this is exactly
        majority vote: the proportion exceeds 0.5 iff more than k/2 of
        the neighbors are class 1. For even k a tie (proportion == 0.5)
        is resolved as class 1, matching the `>=` rule used by
        `logistic_regression.predict_class`; pass a slightly higher
        threshold to flip ties the other way.

        Parameters
        ----------
        X : np.ndarray, shape (M, n_features) or (n_features,)
            Query inputs.
        threshold : float, default 0.5
            Proportions >= threshold are classified as 1, otherwise 0.

        Returns
        -------
        np.ndarray, shape (M,), dtype int
            Predicted class labels in {0, 1}.
        """
        return (self.predict(X) >= threshold).astype(int)
