"""
n-dimensional binary logistic regression trained via gradient descent.

Model
-----
    z      = X @ w + b           (linear score, a.k.a. logit)
    y_hat  = sigmoid(z) = 1 / (1 + exp(-z))

where:
    X     : (N, n) input matrix -- N samples, n features
    w     : (n,)   weight vector -- one weight per feature
    b     : scalar bias term
    z     : (N,)   pre-sigmoid scores
    y_hat : (N,)   predicted probabilities, each in (0, 1)

The linear-regression model y_hat = X @ w + b is unconstrained in (-inf, +inf)
-- great for predicting continuous targets, useless for predicting whether a
sample belongs to class 1. Logistic regression composes that same linear
function with the sigmoid squashing function, mapping any real number to a
probability in (0, 1). The resulting y_hat can then be interpreted as
P(y = 1 | X) under a Bernoulli model of the labels.

Decision boundary
-----------------
The class boundary at threshold 0.5 corresponds exactly to z = 0, i.e. to
the hyperplane X @ w + b = 0. Logistic regression is therefore a *linear*
classifier: the boundary is always a flat hyperplane in feature space.
Curvy boundaries can still be obtained by passing in polynomial / interaction
features (see the companion markdown's section on extensions).
"""

import numpy as np

import ml_from_scratch.classification.loss as loss

LOSS_FUNCTIONS = ["BCE", "Hinge", "MSE"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically-stable elementwise sigmoid.

    The naive form 1 / (1 + exp(-z)) overflows for large negative z (because
    exp(-z) overflows). For large positive z it underflows but the result is
    still ~1, which is fine. We branch on the sign of z and use the
    mathematically-equivalent stable form for each side:

        z >= 0 :  1 / (1 + exp(-z))           -- exp argument is <= 0
        z <  0 :  exp(z) / (1 + exp(z))       -- exp argument is <  0

    Both forms keep the argument to `exp` non-positive, which never overflows.
    """
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg])
    out[neg] = e / (1.0 + e)
    return out


class logistic_regression:
    """
    An n-dimensional binary logistic regression model trained by gradient
    descent.

    Attributes
    ----------
    loss : str
        Name of the loss function being used ("BCE", "Hinge", or "MSE").
    n_features : int
        Number of input features the model expects.
    w : np.ndarray, shape (n_features,)
        Learned weight vector. Initialized to zeros (differs from
        linear_regression, which initializes to ones; see __init__).
    b : float
        Learned bias term. Initialized to 0.
    """

    loss: str
    n_features: int
    w: np.ndarray
    b: float

    def __init__(self, n_features: int, loss: str = "BCE"):
        """
        Parameters
        ----------
        n_features : int
            Number of features in the inputs the model will see. Determines
            the size of the weight vector w.
        loss : str, default "BCE"
            One of "BCE" (binary cross-entropy / log loss), "Hinge"
            (SVM-style margin loss), or "MSE" (mean squared error on
            probabilities, a.k.a. the Brier score).

        Notes
        -----
        w is initialized to zeros rather than ones (the linear_regression
        choice). With w = 0 and b = 0, every initial logit is 0, so every
        initial probability is sigmoid(0) = 0.5 -- a neutral, "I have no
        information" starting point. With w = ones and several features of
        magnitude ~1, the initial logits would be O(n_features) and the
        initial probabilities would saturate near 0 or 1, giving very small
        gradients on the first few steps.
        """
        if loss not in LOSS_FUNCTIONS:
            raise ValueError("Loss must be one of the following:", LOSS_FUNCTIONS)
        if n_features < 1:
            raise ValueError("n_features must be >= 1")

        self.loss = loss
        self.n_features = n_features
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

    def _logits(self, X: np.ndarray) -> np.ndarray:
        """
        Pre-sigmoid scores z = X @ w + b. Internal helper used by both
        predict() and train_one_step() so the linear combination is not
        computed twice per step.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (N, {self.n_features}); got {X.shape}")
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the model to a batch of inputs and return class-1 probabilities.

        For each row x_i of X, the prediction is

            z_i     = x_i . w + b
            y_hat_i = sigmoid(z_i)

        Stacked over N samples this is just

            y_hat = sigmoid(X @ w + b)

        which produces a shape-(N,) vector in a single vectorized step --
        no Python loop over samples is needed.

        Parameters
        ----------
        X : np.ndarray
            Either shape (N, n_features) for a batch of N samples, or
            shape (n_features,) for a single sample (which is reshaped
            to (1, n_features) internally).

        Returns
        -------
        np.ndarray, shape (N,)
            Predicted probabilities P(y = 1 | x_i), each in (0, 1).
        """
        return _sigmoid(self._logits(X))

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Apply the model and return discrete class labels {0, 1} by
        thresholding the predicted probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features) or (n_features,)
            Inputs.
        threshold : float, default 0.5
            Probabilities >= threshold are classified as 1, otherwise 0.
            0.5 is the natural choice when the two classes are equally
            costly to mis-predict; raise it to be more conservative about
            predicting class 1, lower it for the opposite trade-off.

        Returns
        -------
        np.ndarray, shape (N,), dtype int
            Predicted class labels in {0, 1}.
        """
        return (self.predict(X) >= threshold).astype(int)

    def train_one_step(self, alpha: float, X: np.ndarray, y: np.ndarray) -> float:
        """
        Perform one gradient-descent step.

        Computes logits and predictions, evaluates the chosen loss to get
        the gradients (delta_w, delta_b), and then updates the parameters
        with the standard gradient-descent rule:

            w <- w - alpha * delta_w
            b <- b - alpha * delta_b

        The returned error is computed *before* the parameter update -- i.e.
        it reflects the current model, not the updated one.

        Parameters
        ----------
        alpha : float
            Learning rate.
        X : np.ndarray, shape (N, n_features)
            Input feature matrix.
        y : np.ndarray, shape (N,)
            Target labels, each in {0, 1}.

        Returns
        -------
        float
            The loss value evaluated before the parameter update.
        """
        logits = self._logits(X)
        preds = _sigmoid(logits)

        if self.loss == "BCE":
            error, delta_w, delta_b = loss.bce(X, y, logits, preds)
        elif self.loss == "Hinge":
            error, delta_w, delta_b = loss.hinge(X, y, logits, preds)
        elif self.loss == "MSE":
            error, delta_w, delta_b = loss.mse(X, y, logits, preds)

        self.w = self.w - alpha * delta_w
        self.b = self.b - alpha * delta_b

        return error

    def train(
        self,
        alpha: float,
        X: np.ndarray,
        y: np.ndarray,
        max_iterations: int = 100,
        error_change_cutoff: float = 0.001,
    ) -> None:
        """
        Train until either max_iterations is reached, or the change in loss
        between iterations falls below error_change_cutoff.

        Parameters
        ----------
        alpha : float
            Learning rate.
        X : np.ndarray, shape (N, n_features)
            Input feature matrix.
        y : np.ndarray, shape (N,)
            Target labels, each in {0, 1}.
        max_iterations : int, default 100
            Hard cap on the number of gradient-descent steps.
        error_change_cutoff : float, default 0.001
            Early-stopping threshold on |error_t - error_{t-1}|.
        """
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (N, {self.n_features}); got {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(
                f"y must have shape (N,) matching X.shape[0]={X.shape[0]}; "
                f"got {y.shape}"
            )
        unique = np.unique(y)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError(f"y must contain only 0 or 1; got unique values {unique}")

        prev_error = -1
        error = -1
        num_iterations = 0

        # Note: the error reported for each step is the error before the
        # parameters are updated, so the reported error is always "one step
        # behind" the current parameters. For this example, that's fine.
        while prev_error == -1 or num_iterations < max_iterations:
            if num_iterations >= 2 and abs(error - prev_error) < error_change_cutoff:
                break
            num_iterations += 1
            prev_error = error
            error = self.train_one_step(alpha, X, y)

        accuracy = float(np.mean(self.predict_class(X) == y))

        print("Training completed:")
        print("\tIterations:", num_iterations)
        print("\tFinal Error:", error)
        print(f"\tTraining Accuracy: {accuracy:.4f}")
        print(f"\tFinal Weights (w): {self.w}")
        print(f"\tFinal Bias    (b): {self.b}")
