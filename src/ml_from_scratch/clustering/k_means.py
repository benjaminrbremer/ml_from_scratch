"""
K-means clustering.

Model
-----
    Given N unlabeled points X in R^n, partition them into K clusters
    C_1, ..., C_K by choosing K centroids mu_1, ..., mu_K to minimize
    the within-cluster sum of squares (WCSS / inertia):

        J = sum_{k=1}^{K} sum_{i in C_k} || x_i - mu_k ||^2

The algorithm alternates between two steps until convergence:

    Assignment (E-step analogy):
        Assign each point to the nearest centroid:
            label_i = argmin_{k}  || x_i - mu_k ||^2

    Update (M-step analogy):
        Recompute each centroid as the mean of its assigned points:
            mu_k = (1 / |C_k|) * sum_{i in C_k} x_i

Each step monotonically decreases J, and since J is bounded below by 0,
the algorithm is guaranteed to converge (though possibly to a local minimum).

Unlike regression and classification, k-means is *unsupervised* -- there are
no labels y. The "parameters" are the K centroid vectors rather than a weight
vector w and bias b. There is no learning rate because the update step is an
exact analytic minimizer (the mean), not a gradient-descent approximation.

Initialization
--------------
The quality of the final clustering depends heavily on the starting centroids.
Two strategies are supported:

    "random"   : choose K points from X uniformly at random.
    "kmeans++" : the K-means++ seeding scheme (Arthur & Vassilvitskii, 2007).
                 Pick the first centroid uniformly at random, then pick each
                 subsequent centroid with probability proportional to D(x)^2 --
                 the squared distance to the nearest already-chosen centroid.
                 This spreads the initial centroids across the data, reducing
                 the expected WCSS at convergence to O(log K) * OPT.
"""

import numpy as np

INIT_METHODS = ["kmeans++", "random"]


class k_means:
    """
    K-means clustering.

    Attributes
    ----------
    k : int
        Number of clusters.
    n_features : int
        Number of input features the model expects.
    init : str
        Initialization strategy: "kmeans++" (default) or "random".
    random_state : int or None
        Seed for the random number generator. None means non-deterministic.
    centroids : np.ndarray, shape (k, n_features)
        Cluster centroids. Set by fit(); undefined before fit() is called.
    labels_ : np.ndarray, shape (N,)
        Cluster assignment for each point from the last call to fit() or
        predict(). Stored as a convenience so callers don't have to call
        predict() again after fit().
    """

    k: int
    n_features: int
    init: str
    random_state: int | None
    centroids: np.ndarray
    labels_: np.ndarray

    def __init__(
        self,
        k: int,
        n_features: int,
        init: str = "kmeans++",
        random_state: int | None = None,
    ):
        """
        Parameters
        ----------
        k : int
            Number of clusters. Must be >= 1.
        n_features : int
            Number of features in each input vector. Determines the shape of
            the centroid matrix.
        init : str, default "kmeans++"
            Centroid initialization strategy. "kmeans++" spreads the initial
            centroids far apart using D^2 sampling, which substantially
            reduces the chance of a bad local minimum. "random" picks K
            data points uniformly at random -- simpler but more sensitive to
            unlucky starts.
        random_state : int or None, default None
            Seed passed to np.random.default_rng for reproducibility. None
            means a fresh (non-deterministic) RNG each time.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        if init not in INIT_METHODS:
            raise ValueError(f"init must be one of {INIT_METHODS}; got {init!r}")

        self.k = k
        self.n_features = n_features
        self.init = init
        self.random_state = random_state

    def _sq_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Compute squared Euclidean distances from every point to every centroid.

        Broadcasting produces the full (N, K) distance matrix in a single
        vectorized expression -- no Python loop over N samples:

            diff[i, k, j] = X[i, j] - centers[k, j]

        Squaring and summing over the feature axis j gives:

            D[i, k] = sum_j (X[i, j] - centers[k, j])^2 = || x_i - c_k ||^2

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)
        centers : np.ndarray, shape (M, n_features)

        Returns
        -------
        np.ndarray, shape (N, M)
            D[i, k] = squared Euclidean distance from X[i] to centers[k].
        """
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, M, n)
        return np.sum(diff**2, axis=2)  # (N, M)

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Initialize K centroids from the data rows.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)
        rng : np.random.Generator

        Returns
        -------
        np.ndarray, shape (k, n_features)
        """
        n = X.shape[0]
        if self.init == "random":
            indices = rng.choice(n, size=self.k, replace=False)
            return X[indices].copy()

        # K-means++ initialization
        # Step 1: pick the first centroid uniformly at random.
        first = rng.integers(n)
        chosen = [X[first]]

        # Step 2: pick each subsequent centroid with probability proportional
        # to D(x)^2 -- squared distance to the nearest already-chosen centroid.
        for _ in range(self.k - 1):
            centers_so_far = np.array(chosen)  # (m, n_features)
            # sq_dists shape: (N, m); take the min over already-chosen centroids
            sq_dists = self._sq_distances(X, centers_so_far)  # (N, m)
            d2 = sq_dists.min(axis=1)  # (N,) -- distance to nearest centroid

            # Sample proportional to d2; normalize to get a probability vector.
            # If all d2 are zero (degenerate data), fall back to uniform.
            total = d2.sum()
            probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
            next_idx = rng.choice(n, p=probs)
            chosen.append(X[next_idx])

        return np.array(chosen)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample to its nearest centroid.

        This is the assignment (E) step of the k-means algorithm. The
        returned labels are integers in {0, ..., k-1}.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features) or (n_features,)
            Points to assign.

        Returns
        -------
        np.ndarray, shape (N,), dtype int
            labels[i] = index of the centroid closest to X[i].
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (N, {self.n_features}); got {X.shape}")
        sq_dists = self._sq_distances(X, self.centroids)  # (N, k)
        return np.argmin(sq_dists, axis=1)  # (N,)

    def inertia(self, X: np.ndarray) -> float:
        """
        Compute the within-cluster sum of squares (WCSS / inertia).

        J = sum_{i=1}^{N} || x_i - mu_{label_i} ||^2

        This is the objective that k-means minimizes. Lower is better
        (tighter, more compact clusters).

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)

        Returns
        -------
        float
            Sum of squared distances from each point to its assigned centroid.
        """
        labels = self.predict(X)
        sq_dists = self._sq_distances(X, self.centroids)  # (N, k)
        # Pick the distance to each point's assigned centroid.
        return float(sq_dists[np.arange(len(X)), labels].sum())

    def train_one_step(self, X: np.ndarray, tol: float = 1e-4) -> tuple[float, bool]:
        """
        Perform one full k-means iteration: assign then update.

        Assignment (E-step):
            label_i = argmin_k || x_i - mu_k ||^2

        Update (M-step):
            mu_k = mean of all X[i] where label_i == k

        If a cluster ends up empty (can happen with "random" init), its
        centroid is left unchanged.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)
        tol : float, default 1e-4
            Convergence threshold. The step is declared converged when the
            maximum L2 shift of any centroid is smaller than tol.

        Returns
        -------
        inertia : float
            WCSS computed *after* the centroid update (reflects the updated
            model, unlike the regression train_one_step which returns the
            pre-update error).
        converged : bool
            True if the largest centroid shift is < tol.
        """
        labels = self.predict(X)

        new_centroids = np.zeros_like(self.centroids)
        for j in range(self.k):
            assigned = X[labels == j]
            if len(assigned) > 0:
                new_centroids[j] = assigned.mean(axis=0)
            else:
                # Empty cluster: keep the old centroid to avoid a NaN centroid.
                new_centroids[j] = self.centroids[j]

        shifts = np.linalg.norm(new_centroids - self.centroids, axis=1)
        converged = bool(shifts.max() < tol)

        self.centroids = new_centroids
        self.labels_ = self.predict(X)

        return self.inertia(X), converged

    def fit(
        self,
        X: np.ndarray,
        max_iterations: int = 300,
        tol: float = 1e-4,
    ) -> None:
        """
        Fit k-means to X: initialize centroids, then iterate until convergence
        or max_iterations is reached.

        After fitting, self.centroids holds the final cluster centers and
        self.labels_ holds the assignment for every row of X.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)
            Unlabeled data to cluster. N must be >= k.
        max_iterations : int, default 300
            Hard cap on the number of E+M step pairs.
        tol : float, default 1e-4
            Early-stopping threshold: stop when the largest centroid shift
            is smaller than tol (centroids have essentially stopped moving).
        """
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (N, {self.n_features}); got {X.shape}")
        if X.shape[0] < self.k:
            raise ValueError(
                f"Need at least k={self.k} samples to initialize {self.k} "
                f"centroids; got {X.shape[0]}"
            )

        rng = np.random.default_rng(self.random_state)
        self.centroids = self._init_centroids(X, rng)

        inertia = 0.0
        for iteration in range(1, max_iterations + 1):
            inertia, converged = self.train_one_step(X, tol=tol)
            if converged:
                break

        print("Fitting completed:")
        print(f"\tIterations:      {iteration}")
        print(f"\tFinal inertia:   {inertia:.4f}")
        print(
            f"\tCluster sizes:   {[int((self.labels_ == j).sum()) for j in range(self.k)]}"
        )
        print(f"\tCentroids:\n{self.centroids}")
