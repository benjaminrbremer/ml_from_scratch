"""
Gaussian Naive Bayes classifier.

Model
-----
Naive Bayes is a *generative* classifier: instead of directly learning a
boundary between classes (as logistic regression does), it models how each
class *generates* its data, then uses Bayes' theorem to invert: given some
observed x, which class most likely generated it?

    P(y = k | x)  ∝  P(x | y = k) * P(y = k)
        posterior      likelihood     prior

"Gaussian" means we assume each feature is normally distributed within each
class:

    x_j | y = k  ~  N(mu_kj, sigma_kj^2)

"Naive" means we assume features are *conditionally independent* given the
class label. This is almost never exactly true in practice, but the resulting
classifier is surprisingly robust because incorrect probability estimates can
still produce correct argmax decisions (see the companion markdown).

Fitting
-------
Unlike logistic regression, there is no gradient descent. The parameters are
estimated analytically in a single pass over the training data:

    P(y = k)  = N_k / N                    (fraction of class-k samples)
    mu_kj     = mean(x_j for samples in class k)
    sigma_kj^2 = var(x_j for samples in class k)

These are the maximum-likelihood estimates (MLE) for each parameter. No
iterations, no learning rate.

Prediction
----------
For each test sample x, compute the unnormalized log-posterior for every
class and pick the argmax:

    y_hat = argmax_k  [log P(y = k) + sum_j log N(x_j ; mu_kj, sigma_kj^2)]

Working in log-space avoids the float underflow that would occur when
multiplying many small Gaussian PDF values together.
"""

import numpy as np

import ml_from_scratch.classification.naive_bayes.distributions as distributions


class gaussian_naive_bayes:
    """
    Gaussian Naive Bayes classifier supporting binary and multi-class targets.

    Attributes
    ----------
    n_features : int
        Number of input features the model expects.
    var_smoothing : float
        Fraction of the largest observed variance added to every per-class
        variance before storing. Prevents division by zero when a feature has
        zero variance within a class (all identical values). Default 1e-9
        mirrors scikit-learn's GaussianNB.
    classes_ : np.ndarray, shape (K,)
        Sorted array of unique class labels encountered during fit.
    class_log_priors_ : np.ndarray, shape (K,)
        log P(y = k) for each class k, estimated as log(N_k / N).
    means_ : np.ndarray, shape (K, n_features)
        Per-class, per-feature sample means mu_kj.
    variances_ : np.ndarray, shape (K, n_features)
        Per-class, per-feature sample variances sigma_kj^2, with smoothing
        applied.
    """

    n_features: int
    var_smoothing: float
    classes_: np.ndarray
    class_log_priors_: np.ndarray
    means_: np.ndarray
    variances_: np.ndarray

    def __init__(self, n_features: int, var_smoothing: float = 1e-9):
        """
        Parameters
        ----------
        n_features : int
            Number of features in the inputs the model will see.
        var_smoothing : float, default 1e-9
            Stability parameter. The value added to each per-class variance is

                var_smoothing * max(variances_)

            where max(variances_) is taken over all classes and features after
            the MLE fit. Setting this to zero disables smoothing entirely;
            raise it if predictions are numerically unstable on very low-
            variance features.

        Notes
        -----
        The model is not fitted until fit() is called. Calling predict before
        fit will raise an AttributeError on classes_ / means_ / variances_.
        """
        if n_features < 1:
            raise ValueError("n_features must be >= 1")

        self.n_features = n_features
        self.var_smoothing = var_smoothing

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        """Reshape single samples and validate feature count."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (N, {self.n_features}); got {X.shape}")
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "gaussian_naive_bayes":
        """
        Fit the model by computing class priors, per-class means, and per-class
        variances from the training data. No iteration — one closed-form pass.

        For each class k:

            N_k       = number of samples where y == k
            P(y = k)  = N_k / N
            mu_kj     = (1 / N_k) * sum_{i: y_i = k} x_ij
            sigma_kj^2 = (1 / N_k) * sum_{i: y_i = k} (x_ij - mu_kj)^2

        These are the maximum-likelihood estimates. See the companion markdown
        for the derivation from first principles.

        After computing the raw MLE variances, variance smoothing is applied:

            sigma_kj^2  <-  sigma_kj^2  +  var_smoothing * max(sigma^2)

        where max(sigma^2) is taken over all (k, j) entries. This prevents
        division by zero in gaussian_log_likelihood when a feature is
        constant within a class.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)
            Training feature matrix.
        y : np.ndarray, shape (N,)
            Training class labels. Any integer or string labels are accepted;
            they are stored in sorted order in classes_.

        Returns
        -------
        self : gaussian_naive_bayes
            Returns self to allow chaining: nb.fit(X, y).predict(X_test).
        """
        X = self._validate_X(X)
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(
                f"y must have shape (N,) matching X.shape[0]={X.shape[0]}; got {y.shape}"
            )

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = X.shape[0]

        self.means_ = np.zeros((n_classes, self.n_features))
        self.variances_ = np.zeros((n_classes, self.n_features))
        self.class_log_priors_ = np.zeros(n_classes)

        for k, cls in enumerate(self.classes_):
            X_k = X[y == cls]  # samples belonging to class k
            self.means_[k] = X_k.mean(axis=0)
            self.variances_[k] = X_k.var(axis=0)
            self.class_log_priors_[k] = np.log(X_k.shape[0] / n_samples)

        # Variance smoothing: add a small fraction of the global max variance
        # to every entry. Prevents log(0) / division-by-zero in the likelihood.
        self.variances_ += self.var_smoothing * self.variances_.max()

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Unnormalized log-posterior log P(y = k | x) for every sample and class.

        By Bayes' theorem (dropping the evidence P(x), which is constant
        across classes):

            log P(y = k | x) = log P(x | y = k) + log P(y = k) + const

        The log-likelihood log P(x | y = k) is computed by
        distributions.gaussian_log_likelihood. Adding the log-prior broadcasts
        correctly because class_log_priors_ has shape (K,).

        The result is *unnormalized* (does not sum to 1 in probability space).
        Use predict_proba for calibrated probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features) or (n_features,)

        Returns
        -------
        np.ndarray, shape (N, K)
            log P(y = k | x_i) up to an additive constant (the log evidence).
        """
        X = self._validate_X(X)
        log_ll = distributions.gaussian_log_likelihood(X, self.means_, self.variances_)
        return log_ll + self.class_log_priors_  # (N, K)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Normalized posterior probabilities P(y = k | x) for every sample.

        Converts log-posteriors to probabilities using the log-sum-exp trick:

            log Z_i = log sum_k exp(log_post_ik)

            P(y = k | x_i) = exp(log_post_ik - log Z_i)

        Subtracting log Z_i before exponentiating keeps the values in a
        numerically safe range. The subtraction does not change the argmax, so
        predict() is equivalent whether it calls predict_log_proba or
        predict_proba.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features) or (n_features,)

        Returns
        -------
        np.ndarray, shape (N, K)
            Posterior probabilities; each row sums to 1.
        """
        log_post = self.predict_log_proba(X)  # (N, K)
        log_Z = np.log(
            np.exp(log_post - log_post.max(axis=1, keepdims=True)).sum(
                axis=1, keepdims=True
            )
        )
        return np.exp(log_post - log_post.max(axis=1, keepdims=True) - log_Z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for each sample.

        Returns the class with the highest log-posterior. Because log is
        monotone, this is identical to argmax over the posterior probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features) or (n_features,)

        Returns
        -------
        np.ndarray, shape (N,)
            Predicted class labels drawn from classes_.
        """
        log_post = self.predict_log_proba(X)  # (N, K)
        return self.classes_[np.argmax(log_post, axis=1)]
