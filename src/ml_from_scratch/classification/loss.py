"""
Loss functions for n-dimensional binary logistic regression.

Each function takes:
    X      : (N, n) matrix of inputs    -- N samples, n features
    y      : (N,)   vector of targets   -- values in {0, 1}
    logits : (N,)   pre-sigmoid scores  -- z = X @ w + b
    preds  : (N,)   probabilities       -- p = sigmoid(z), in (0, 1)

and returns:
    error    : scalar value of the loss
    delta_w  : (n,) gradient of the loss with respect to the weight vector w
    delta_b  : scalar gradient of the loss with respect to the bias b

Why pass both logits and probabilities
--------------------------------------
The linear-regression loss module passes only `preds`, because in linear
regression the model's natural output (X @ w + b) is exactly what every loss
operates on. In logistic regression, different losses want different views of
the same underlying linear score z = X @ w + b:

    * BCE and Brier (MSE on probabilities) operate on p = sigmoid(z).
    * Hinge operates on the raw score z itself.

Recovering z from p via logit(p) = log(p / (1 - p)) is numerically unstable
when p is close to 0 or 1 (it blows up to +/- infinity). Recomputing
sigmoid(z) inside every loss would just duplicate work. So we pass both --
each loss uses whichever quantity is natural and ignores the other.

Conventions used throughout
---------------------------
    * Residuals are defined as  r = preds - y, shape (N,). Note the sign is
      flipped from the linear-regression convention (which used y - preds);
      this is so the BCE gradient comes out as +(1/N) * X.T @ r without an
      extra minus sign, matching the textbook form (preds - y).
    * "X.T @ v" is the vectorized form of the per-sample sum sum_i x_i * v_i,
      where x_i is the i-th row of X. X.T has shape (n, N), v has shape (N,),
      so the result is shape (n,) -- one gradient entry per feature. No
      Python loop over samples is needed.
    * np.mean(v) is equivalent to (1/N) * np.sum(v) and is used to make the
      "averaged over N samples" structure of each loss explicit.
"""

import numpy as np

# Clip applied to probabilities before taking log in BCE, to avoid log(0).
# The gradient form (preds - y) is already finite and does not need clipping;
# this clip only affects the reported error value.
_EPS = 1e-12


def bce(
    X: np.ndarray,
    y: np.ndarray,
    logits: np.ndarray,
    preds: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """
    Binary Cross-Entropy (a.k.a. log loss) and its gradients.

    Loss
    ----
        L(w, b) = -(1/N) * sum_i [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]

    where p_i = sigmoid(z_i) and z_i = x_i . w + b.

    Gradient derivation
    -------------------
        Treat L as a function of z first, then chain through z = X @ w + b.

        For one sample:
            l_i = -[ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
            dp/dz = p * (1 - p)                    (sigmoid derivative)
            dl_i / dp_i = -y_i / p_i + (1 - y_i) / (1 - p_i)
                        = (p_i - y_i) / (p_i * (1 - p_i))

            dl_i / dz_i = (dl_i / dp_i) * (dp_i / dz_i)
                        = (p_i - y_i) / (p_i * (1 - p_i)) * p_i * (1 - p_i)
                        = p_i - y_i

        That cancellation is the famous "BCE + sigmoid gives a clean gradient"
        result. Averaging over N samples:

            dL/dz_i = (p_i - y_i) / N

        Now chain through z_i = x_i . w + b:
            dz_i / dw_j = x_ij
            dz_i / db   = 1

            dL/dw_j = (1/N) * sum_i (p_i - y_i) * x_ij
                    = (1/N) * (X.T @ (preds - y))_j
            dL/db   = (1/N) * sum_i (p_i - y_i)

    Notice the structural parallel to MSE in linear regression: replace the
    linear residual (preds - y) interpretation accordingly and the
    "delta_w = (1/N) * X.T @ residual" pattern is the same.

    Parameters
    ----------
    X : np.ndarray, shape (N, n)
        Input feature matrix.
    y : np.ndarray, shape (N,)
        Target values, each in {0, 1}.
    logits : np.ndarray, shape (N,)
        Pre-sigmoid scores. Unused by BCE -- accepted for signature uniformity.
    preds : np.ndarray, shape (N,)
        Sigmoid probabilities, in (0, 1).

    Returns
    -------
    error : float
        The BCE value.
    delta_w : np.ndarray, shape (n,)
        Gradient with respect to the weight vector.
    delta_b : float
        Gradient with respect to the bias.
    """
    del logits  # not needed; documented in the signature
    n_samples = X.shape[0]

    # Clip only for the error value; the gradient is numerically fine.
    p_clipped = np.clip(preds, _EPS, 1.0 - _EPS)
    error = float(-np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped)))

    r = preds - y  # shape (N,)
    delta_w = (1.0 / n_samples) * (X.T @ r)
    delta_b = (1.0 / n_samples) * float(np.sum(r))

    return error, delta_w, delta_b


def hinge(
    X: np.ndarray,
    y: np.ndarray,
    logits: np.ndarray,
    preds: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """
    Hinge loss (the SVM loss) and its (sub)gradients.

    Hinge is most naturally defined with labels in {-1, +1}. We convert the
    incoming y in {0, 1} via  y_pm = 2*y - 1.

    Loss
    ----
        L(w, b) = (1/N) * sum_i max(0, 1 - y_pm_i * z_i)

    where z_i = x_i . w + b is the raw linear score (no sigmoid). The loss is
    zero whenever the model's score has the correct sign and magnitude
    >= 1 (i.e. the sample is on the right side of the margin); otherwise it
    grows linearly in z.

    Gradient derivation
    -------------------
        Let m_i = 1 - y_pm_i * z_i be the per-sample "margin violation".
        Define the active set indicator:
            a_i = 1  if m_i > 0   (sample inside / past the margin)
            a_i = 0  otherwise    (sample correctly classified with margin)

        Then max(0, m_i) is differentiable everywhere except at m_i = 0, and:
            d max(0, m_i) / d m_i = a_i
            d m_i / d z_i = -y_pm_i

        Chain rule:
            dL/dz_i = -(1/N) * y_pm_i * a_i

        Through z_i = x_i . w + b:
            dL/dw_j = -(1/N) * sum_i x_ij * y_pm_i * a_i
                    = -(1/N) * (X.T @ (y_pm * a))_j
            dL/db   = -(1/N) * sum_i y_pm_i * a_i

        At m_i = 0 the function is non-differentiable; we pick the subgradient
        a_i = 0 (matches `(m > 0)` in code), a standard choice.

    Parameters
    ----------
    X : np.ndarray, shape (N, n)
        Input feature matrix.
    y : np.ndarray, shape (N,)
        Target values, each in {0, 1}.
    logits : np.ndarray, shape (N,)
        Pre-sigmoid raw scores z = X @ w + b. This is what hinge uses.
    preds : np.ndarray, shape (N,)
        Sigmoid probabilities. Unused by hinge -- accepted for signature
        uniformity.

    Returns
    -------
    error : float
        The hinge loss value.
    delta_w : np.ndarray, shape (n,)
        Gradient with respect to the weight vector.
    delta_b : float
        Gradient with respect to the bias.
    """
    del preds  # not needed; documented in the signature
    n_samples = X.shape[0]

    y_pm = 2.0 * y - 1.0  # {0, 1} -> {-1, +1}
    margins = 1.0 - y_pm * logits  # shape (N,)
    active = (margins > 0.0).astype(float)  # subgradient: 0 at the kink

    error = float(np.mean(np.maximum(0.0, margins)))

    g = y_pm * active  # shape (N,)
    delta_w = -(1.0 / n_samples) * (X.T @ g)
    delta_b = -(1.0 / n_samples) * float(np.sum(g))

    return error, delta_w, delta_b


def mse(
    X: np.ndarray,
    y: np.ndarray,
    logits: np.ndarray,
    preds: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """
    Mean Squared Error on probabilities (a.k.a. the Brier score) and its
    gradients.

    Loss
    ----
        L(w, b) = (1/N) * sum_i (y_i - p_i)^2

    where p_i = sigmoid(z_i). Unlike BCE, MSE-on-probabilities is *not*
    convex in (w, b), which makes it a less reliable training loss in
    practice. It is included here mainly for comparison: the same quantity
    that worked well as a loss for linear regression becomes a worse choice
    once a sigmoid sits between the parameters and the prediction.

    Gradient derivation
    -------------------
        Let r_i = y_i - p_i.

            dL/dp_i = -(2/N) * r_i
            dp_i/dz_i = p_i * (1 - p_i)      (sigmoid derivative)
            dL/dz_i = -(2/N) * r_i * p_i * (1 - p_i)

        Through z_i = x_i . w + b:
            dL/dw_j = -(2/N) * sum_i x_ij * r_i * p_i * (1 - p_i)
                    = -(2/N) * (X.T @ (r * p * (1 - p)))_j
            dL/db   = -(2/N) * sum_i r_i * p_i * (1 - p_i)

    Compared to BCE, the gradient has an extra `p * (1 - p)` factor that
    vanishes whenever the model is very confident (p near 0 or 1) -- even
    when the prediction is *confidently wrong*. This is the "vanishing
    gradient" issue that makes MSE a poor choice for classifiers.

    Parameters
    ----------
    X : np.ndarray, shape (N, n)
        Input feature matrix.
    y : np.ndarray, shape (N,)
        Target values, each in {0, 1}.
    logits : np.ndarray, shape (N,)
        Pre-sigmoid scores. Unused by Brier -- accepted for signature
        uniformity.
    preds : np.ndarray, shape (N,)
        Sigmoid probabilities, in (0, 1).

    Returns
    -------
    error : float
        The Brier score.
    delta_w : np.ndarray, shape (n,)
        Gradient with respect to the weight vector.
    delta_b : float
        Gradient with respect to the bias.
    """
    del logits  # not needed; documented in the signature
    n_samples = X.shape[0]

    r = y - preds
    error = float(np.mean(r**2))

    g = r * preds * (1.0 - preds)  # shape (N,)
    delta_w = -(2.0 / n_samples) * (X.T @ g)
    delta_b = -(2.0 / n_samples) * float(np.sum(g))

    return error, delta_w, delta_b
