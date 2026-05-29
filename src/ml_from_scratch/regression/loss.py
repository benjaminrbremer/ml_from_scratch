"""
Loss functions for n-dimensional linear regression.

Each function takes:
    X      : (N, n) matrix of inputs   -- N samples, n features
    y      : (N,)   vector of targets
    preds  : (N,)   vector of model predictions (y_hat)

and returns:
    error    : scalar value of the loss
    delta_w  : (n,) gradient of the loss with respect to the weight vector w
    delta_b  : scalar gradient of the loss with respect to the bias b

Conventions used throughout:
    * Residuals are defined as  r = y - preds, shape (N,).
    * "X.T @ r" is the vectorized form of the per-sample sum  sum_i x_i * r_i,
      where x_i is the i-th row of X. NumPy's "@" operator is matrix
      multiplication; X.T has shape (n, N), r has shape (N,), so the
      result is shape (n,) -- one gradient entry per feature.
    * np.mean(v) is equivalent to (1/N) * np.sum(v) and is used to make the
      "averaged over N samples" structure of each loss explicit.
"""

import numpy as np


def mse(
    X: np.ndarray, y: np.ndarray, preds: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """
    Mean Squared Error and its gradients.

    Loss
    ----
        L(w, b) = (1/N) * sum_i (y_i - y_hat_i)^2
                = (1/N) * sum_i r_i^2

    Gradient derivation
    -------------------
        y_hat_i = x_i . w + b
        d r_i / d w_j = -x_ij
        d r_i / d b   = -1

        dL/dw_j = (1/N) * sum_i 2 * r_i * (d r_i / d w_j)
                = -(2/N) * sum_i r_i * x_ij
                = -(2/N) * (X.T @ r)_j

        dL/db   = (1/N) * sum_i 2 * r_i * (d r_i / d b)
                = -(2/N) * sum_i r_i

    Parameters
    ----------
    X : np.ndarray, shape (N, n)
        Input feature matrix.
    y : np.ndarray, shape (N,)
        Target values.
    preds : np.ndarray, shape (N,)
        Model predictions.

    Returns
    -------
    error : float
        The MSE value.
    delta_w : np.ndarray, shape (n,)
        Gradient with respect to the weight vector.
    delta_b : float
        Gradient with respect to the bias.
    """
    n_samples = X.shape[0]
    r = y - preds

    error = float(np.mean(r**2))
    delta_w = -(2.0 / n_samples) * (X.T @ r)
    delta_b = -(2.0 / n_samples) * float(np.sum(r))

    return error, delta_w, delta_b


def rmse(
    X: np.ndarray, y: np.ndarray, preds: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """
    Root Mean Squared Error and its gradients.

    Loss
    ----
        L(w, b) = sqrt( (1/N) * sum_i r_i^2 )  =  sqrt(MSE)

    Gradient derivation
    -------------------
        Let S = (1/N) * sum_i r_i^2, so L = sqrt(S).
        By the chain rule,  dL/dtheta = (1 / (2 * sqrt(S))) * dS/dtheta
                                      = (1 / (2 * L))       * dS/dtheta.

        We already know from the MSE derivation:
            dS/dw_j = -(2/N) * sum_i r_i * x_ij
            dS/db   = -(2/N) * sum_i r_i

        Substituting and simplifying the factor of 2:
            dL/dw_j = -(1 / (N * L)) * sum_i r_i * x_ij
                    = -(1 / (N * L)) * (X.T @ r)_j
            dL/db   = -(1 / (N * L)) * sum_i r_i

    Parameters
    ----------
    X : np.ndarray, shape (N, n)
        Input feature matrix.
    y : np.ndarray, shape (N,)
        Target values.
    preds : np.ndarray, shape (N,)
        Model predictions.

    Returns
    -------
    error : float
        The RMSE value.
    delta_w : np.ndarray, shape (n,)
        Gradient with respect to the weight vector.
    delta_b : float
        Gradient with respect to the bias.
    """
    n_samples = X.shape[0]
    r = y - preds

    error = float(np.sqrt(np.mean(r**2)))
    delta_w = -(1.0 / (n_samples * error)) * (X.T @ r)
    delta_b = -(1.0 / (n_samples * error)) * float(np.sum(r))

    return error, delta_w, delta_b


def mae(
    X: np.ndarray, y: np.ndarray, preds: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """
    Mean Absolute Error and its gradients.

    Loss
    ----
        L(w, b) = (1/N) * sum_i |y_i - y_hat_i|
                = (1/N) * sum_i |r_i|

    Gradient derivation
    -------------------
        d|r| / dr = sign(r)     (subgradient: sign(0) := 0)
        d r_i / d w_j = -x_ij
        d r_i / d b   = -1

        dL/dw_j = (1/N) * sum_i sign(r_i) * (-x_ij)
                = -(1/N) * sum_i x_ij * sign(r_i)
                = -(1/N) * (X.T @ sign(r))_j

        dL/db   = -(1/N) * sum_i sign(r_i)

        Note on differentiability: |x| is not differentiable at 0. We pick
        the subgradient sign(0) = 0, which is what np.sign returns by default.

    Parameters
    ----------
    X : np.ndarray, shape (N, n)
        Input feature matrix.
    y : np.ndarray, shape (N,)
        Target values.
    preds : np.ndarray, shape (N,)
        Model predictions.

    Returns
    -------
    error : float
        The MAE value.
    delta_w : np.ndarray, shape (n,)
        Gradient with respect to the weight vector.
    delta_b : float
        Gradient with respect to the bias.
    """
    n_samples = X.shape[0]
    r = y - preds
    s = np.sign(r)

    error = float(np.mean(np.abs(r)))
    delta_w = -(1.0 / n_samples) * (X.T @ s)
    delta_b = -(1.0 / n_samples) * float(np.sum(s))

    return error, delta_w, delta_b
