"""
Binary decision tree trained via recursive greedy splitting (CART).

Model
-----
A decision tree partitions feature space into axis-aligned rectangular regions
by repeatedly splitting on a single feature at a single threshold. The tree is
represented as a binary tree of nodes:

    * Internal nodes store (feature_idx, threshold): route samples left if
      X[feature_idx] <= threshold, right otherwise.
    * Leaf nodes store class_probs: the fraction of training samples of each
      class that fell into this leaf during training.

At prediction time, each sample walks from the root to a leaf by following the
left/right branches at each internal node, and the leaf's class_probs[1] is
returned as the predicted probability of class 1.

How this differs from linear / logistic regression
----------------------------------------------------
Linear and logistic regression find model parameters (w, b) by minimizing a
differentiable loss with gradient descent — an iterative, incremental process.
Decision trees have no parameters in that sense: training is a single pass
that builds the tree top-down by recursively finding the split that best
reduces an impurity measure (entropy or Gini). There is no learning rate, no
gradient, and no concept of "one training step". The entire tree is built in
one call to `train`.

Because splits are axis-aligned (feature j <= threshold), decision boundaries
are piecewise-constant "staircase" shapes, unlike the flat hyperplane of
logistic regression. A tree of depth d can approximate arbitrarily complex
boundaries — at the cost of overfitting when d is too large.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import ml_from_scratch.classification.trees.impurity as impurity

CRITERIA = impurity.CRITERIA


@dataclass
class _Node:
    """
    A single node in the decision tree.

    Leaf nodes have class_probs set and feature_idx / threshold / left / right
    all None. Internal nodes have feature_idx, threshold, left, and right set,
    and class_probs is None.
    """

    feature_idx: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None
    class_probs: np.ndarray | None = None
    n_samples: int = 0


class decision_tree:
    """
    A binary decision tree for binary classification.

    The tree is built with the CART (Classification and Regression Trees)
    algorithm: at each node, the feature and threshold that maximise the
    impurity gain are chosen greedily. Recursion stops when any of the
    following stopping conditions is met:

        1. The current depth equals max_depth.
        2. The node has fewer than min_samples_split samples.
        3. All samples in the node share the same class (pure node; gain = 0).
        4. No split improves impurity (e.g. all features are constant).

    Attributes
    ----------
    criterion : str
        Impurity measure. "entropy" (information gain) or "gini" (Gini gain).
    max_depth : int or None
        Maximum depth of the tree. None means grow until purity or until
        min_samples_split prevents further splits. Shallow trees underfit;
        deep trees overfit.
    min_samples_split : int
        Minimum number of samples required to attempt a split. Acts as an
        implicit regulariser: larger values produce shallower trees.
    root : _Node or None
        Root of the fitted tree. None before train() is called.
    """

    criterion: str
    max_depth: int | None
    min_samples_split: int
    root: _Node | None

    def __init__(
        self,
        criterion: str = "entropy",
        max_depth: int | None = None,
        min_samples_split: int = 2,
    ):
        """
        Parameters
        ----------
        criterion : str, default "entropy"
            Impurity criterion used to evaluate candidate splits.
            "entropy" uses information gain (Shannon entropy, base-2 bits).
            "gini" uses Gini gain. Both produce nearly identical trees in
            practice; entropy is the classic information-theoretic choice.
        max_depth : int or None, default None
            Maximum depth allowed. None = grow until stopping conditions
            (purity or min_samples_split) prevent further splits. Controlling
            depth is the primary regularisation knob for decision trees.
        min_samples_split : int, default 2
            A node must have at least this many samples to be split. Nodes
            with fewer samples become leaves regardless of impurity.
        """
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}; got {criterion!r}")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1 or None")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build the decision tree from training data.

        Unlike gradient-descent models, training is a single recursive pass
        that constructs the tree top-down. There is no learning rate and no
        concept of "one step" — the entire tree is built by this call.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)
            Input feature matrix. N samples, each with n_features features.
        y : np.ndarray, shape (N,)
            Binary integer class labels, each in {0, 1}.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, n_features); got shape {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(
                f"y must have shape (N,) matching X.shape[0]={X.shape[0]}; got {y.shape}"
            )
        unique = np.unique(y)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError(f"y must contain only 0 or 1; got unique values {unique}")

        self.root = self._build_tree(X, y, depth=0)

        n_leaves = self._count_leaves(self.root)
        print("Training completed:")
        print(f"\tLeaves: {n_leaves}")
        print(f"\tDepth:  {self._tree_depth(self.root)}")
        train_acc = float(np.mean(self.predict_class(X) == y))
        print(f"\tTraining Accuracy: {train_acc:.4f}")

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """
        Recursively build the tree from the subset (X, y) at the current depth.

        Returns a leaf node if any stopping condition is met, otherwise returns
        an internal node whose left and right children are built recursively.
        """
        n_samples = len(y)
        node = _Node(n_samples=n_samples)

        # --- stopping conditions ---
        at_max_depth = self.max_depth is not None and depth >= self.max_depth
        too_few = n_samples < self.min_samples_split
        is_pure = len(np.unique(y)) == 1

        if at_max_depth or too_few or is_pure:
            node.class_probs = np.bincount(y, minlength=2) / n_samples
            return node

        # --- find the best split ---
        feat, threshold, gain = impurity.best_split(X, y, self.criterion)

        if gain == 0.0:
            # No split improves impurity — make a leaf
            node.class_probs = np.bincount(y, minlength=2) / n_samples
            return node

        # --- split and recurse ---
        mask = X[:, feat] <= threshold
        node.feature_idx = feat
        node.threshold = threshold
        node.left = self._build_tree(X[mask], y[mask], depth + 1)
        node.right = self._build_tree(X[~mask], y[~mask], depth + 1)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class-1 probability for each sample in X.

        Each sample walks from the root to a leaf by following left/right
        branches (left if X[feat] <= threshold, right otherwise). The leaf's
        stored class_probs[1] is the predicted probability.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features) or (n_features,)
            Input feature matrix. A 1-D input is treated as a single sample.

        Returns
        -------
        np.ndarray, shape (N,)
            Predicted probabilities for class 1, each in [0, 1].
        """
        if self.root is None:
            raise RuntimeError("Call train() before predict().")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._traverse(self.root, x)[1] for x in X])

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict discrete class labels {0, 1} by thresholding predict().

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features) or (n_features,)
            Inputs.
        threshold : float, default 0.5
            Samples with predicted class-1 probability >= threshold are
            classified as 1. The default 0.5 minimises expected error under
            equal class costs. Raise it to be more conservative about
            predicting class 1.

        Returns
        -------
        np.ndarray, shape (N,), dtype int
            Predicted class labels in {0, 1}.
        """
        return (self.predict(X) >= threshold).astype(int)

    def _traverse(self, node: _Node, x: np.ndarray) -> np.ndarray:
        """
        Walk a single sample x from the given node to a leaf.

        Returns the leaf's class_probs array, shape (2,).
        """
        if node.class_probs is not None:
            return node.class_probs
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(node.left, x)
        return self._traverse(node.right, x)

    def _count_leaves(self, node: _Node | None) -> int:
        if node is None:
            return 0
        if node.class_probs is not None:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _tree_depth(self, node: _Node | None) -> int:
        if node is None or node.class_probs is not None:
            return 0
        return 1 + max(self._tree_depth(node.left), self._tree_depth(node.right))
