"""
Impurity measures and split-selection for decision-tree binary classification.

Each impurity function takes a 1-D integer label array y (values in {0, 1})
and returns a scalar impurity score. Lower is purer (less uncertainty).

The two standard measures
--------------------------
    entropy(y) = -sum_k  p_k * log2(p_k)
    gini(y)    = 1 - sum_k p_k^2

where the sum is over the K classes present (K = 2 here) and p_k is the
fraction of samples in class k. Both are zero for a pure node (only one class)
and maximised at 0.5 / 0.5 class balance.

Information gain and Gini gain
-------------------------------
At each candidate split a parent node S is divided into a left child S_L and
a right child S_R. The gain measures how much the split reduces impurity:

    IG(S, S_L, S_R) = H(S) - (|S_L|/|S|) * H(S_L) - (|S_R|/|S|) * H(S_R)

where H is either entropy or Gini. The weighted average of child impurities is
the expected impurity *after* the split; subtracting it from the parent's
impurity gives the reduction. CART picks the split that maximises this gain.

Finding the best split
-----------------------
best_split(X, y, criterion) iterates over every feature and every candidate
threshold (midpoint between adjacent distinct values), evaluates the gain, and
returns the feature index and threshold that yield the highest gain.

Candidate thresholds are midpoints between sorted unique values: if feature j
has unique values [1, 3, 7], the candidates are [2, 5] -- enough to identify
every possible ordering-based split, but not redundant.
"""

import numpy as np

CRITERIA = ["entropy", "gini"]


def entropy(y: np.ndarray) -> float:
    """
    Shannon entropy (base-2) of a binary label array.

    Definition
    ----------
        H(y) = -sum_k p_k * log2(p_k)

    where p_k = count(y == k) / len(y) and the sum is over classes with
    p_k > 0 (to avoid 0 * log(0), which is defined as 0 by convention).

    Properties
    ----------
    * H = 0 when all samples share one class (pure node).
    * H = 1 when classes are perfectly balanced (maximum uncertainty for
      binary classification, base-2 units = bits).
    * H is concave in the class probabilities, which is what makes the
      weighted-average formulation in information_gain a valid guarantee that
      splitting can only reduce or maintain entropy, never increase it.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Integer class labels, each in {0, 1}.

    Returns
    -------
    float
        Entropy in bits. 0.0 for a pure node.
    """
    n = len(y)
    if n == 0:
        return 0.0
    counts = np.bincount(y, minlength=2)
    probs = counts / n
    # 0 * log(0) is conventionally 0; mask out zero probabilities
    mask = probs > 0
    return float(-np.sum(probs[mask] * np.log2(probs[mask])))


def gini(y: np.ndarray) -> float:
    """
    Gini impurity of a binary label array.

    Definition
    ----------
        G(y) = 1 - sum_k p_k^2

    where p_k is the fraction of samples in class k.

    Gini vs entropy
    ---------------
    Both measures are maximised at perfect class balance (G_max = 0.5,
    H_max = 1.0) and equal 0 at purity. In practice they produce nearly
    identical splits for binary classification: their Taylor expansions around
    p = 0.5 agree to second order. Gini is slightly cheaper to compute (no log)
    and is the default in scikit-learn's CART implementation; entropy is the
    classic formulation from information theory.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Integer class labels, each in {0, 1}.

    Returns
    -------
    float
        Gini impurity. 0.0 for a pure node, 0.5 at maximum.
    """
    n = len(y)
    if n == 0:
        return 0.0
    counts = np.bincount(y, minlength=2)
    probs = counts / n
    return float(1.0 - np.sum(probs**2))


def information_gain(
    y: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray,
) -> float:
    """
    Information gain of a split using Shannon entropy.

    Definition
    ----------
        IG(S, S_L, S_R) = H(S) - (|S_L|/|S|) * H(S_L) - (|S_R|/|S|) * H(S_R)

    The weighted average of child entropies equals the *expected* entropy after
    the split. Subtracting it from the parent entropy gives the expected
    reduction in uncertainty. Because entropy is concave, any non-trivial split
    of a mixed node will yield IG >= 0.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Parent node labels.
    y_left : np.ndarray, shape (N_L,)
        Labels routed to the left child.
    y_right : np.ndarray, shape (N_R,)
        Labels routed to the right child.

    Returns
    -------
    float
        Information gain. 0.0 when the split does not reduce entropy.
    """
    n = len(y)
    if n == 0:
        return 0.0
    weight_left = len(y_left) / n
    weight_right = len(y_right) / n
    return entropy(y) - weight_left * entropy(y_left) - weight_right * entropy(y_right)


def gini_gain(
    y: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray,
) -> float:
    """
    Gain of a split using Gini impurity (Gini gain, a.k.a. Gini reduction).

    Definition
    ----------
        GG(S, S_L, S_R) = G(S) - (|S_L|/|S|) * G(S_L) - (|S_R|/|S|) * G(S_R)

    Structurally identical to information_gain but using Gini rather than
    entropy. In practice both criteria produce nearly the same tree.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Parent node labels.
    y_left : np.ndarray, shape (N_L,)
        Labels routed to the left child.
    y_right : np.ndarray, shape (N_R,)
        Labels routed to the right child.

    Returns
    -------
    float
        Gini gain. 0.0 when the split does not reduce Gini impurity.
    """
    n = len(y)
    if n == 0:
        return 0.0
    weight_left = len(y_left) / n
    weight_right = len(y_right) / n
    return gini(y) - weight_left * gini(y_left) - weight_right * gini(y_right)


def best_split(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "entropy",
) -> tuple[int, float, float]:
    """
    Find the feature and threshold that maximise impurity gain.

    Algorithm
    ---------
    For each feature j in 0..n_features-1:
      1. Sort the unique values of X[:, j].
      2. For each adjacent pair (v_k, v_{k+1}), try threshold = (v_k + v_{k+1}) / 2.
      3. Split: left = samples with X[:, j] <= threshold, right = the rest.
      4. Compute gain(y, y_left, y_right) using the chosen criterion.
      5. Keep track of the (j, threshold) pair with the highest gain.

    Using midpoints between unique values is equivalent to testing all possible
    ordered splits on that feature, but avoids redundant thresholds (two
    thresholds between the same pair of consecutive distinct values give
    identical splits).

    If no split improves on the parent impurity (e.g. the node is already pure,
    or every feature is constant), returns (-1, nan, 0.0) as the sentinel.

    Parameters
    ----------
    X : np.ndarray, shape (N, n_features)
        Input feature matrix.
    y : np.ndarray, shape (N,)
        Integer class labels, each in {0, 1}.
    criterion : str, default "entropy"
        Impurity criterion. One of "entropy" (information gain) or "gini"
        (Gini gain).

    Returns
    -------
    best_feature : int
        Index of the feature to split on. -1 if no improving split exists.
    best_threshold : float
        Threshold value. np.nan if no improving split exists.
    best_gain : float
        Gain of the best split. 0.0 if no improving split exists.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}; got {criterion!r}")

    gain_fn = information_gain if criterion == "entropy" else gini_gain

    n_features = X.shape[1]
    best_feature = -1
    best_threshold = float("nan")
    best_gain = 0.0

    for j in range(n_features):
        unique_vals = np.unique(X[:, j])
        if len(unique_vals) < 2:
            continue  # constant feature — no split possible

        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for threshold in thresholds:
            mask_left = X[:, j] <= threshold
            y_left = y[mask_left]
            y_right = y[~mask_left]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            gain = gain_fn(y, y_left, y_right)
            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_threshold = threshold

    return best_feature, best_threshold, best_gain
