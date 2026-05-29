"""
n-dimensional linear regression trained via gradient descent.

Model
-----
    y_hat = X @ w + b

where:
    X : (N, n) input matrix -- N samples, n features
    w : (n,)   weight vector -- one weight per feature
    b : scalar bias term
    y_hat : (N,) vector of predictions

This is the natural generalization of 2D linear regression (y = m*x + b)
to n features: the scalar slope m becomes a weight vector w, and the
input scalar x becomes a feature vector x_i for each sample. The
prediction for one sample is the dot product x_i . w plus the bias b;
stacking samples row-wise into X turns that into the single matrix
expression X @ w + b.
"""

import numpy as np

import ml_from_scratch.regression.loss as loss

LOSS_FUNCTIONS = ["MSE", "RMSE", "MAE"]


class linear_regression:
    """
    An n-dimensional linear regression model trained by gradient descent.

    Attributes
    ----------
    loss : str
        Name of the loss function being used ("MSE", "RMSE", or "MAE").
    n_features : int
        Number of input features the model expects.
    w : np.ndarray, shape (n_features,)
        Learned weight vector. Initialized to ones.
    b : float
        Learned bias term. Initialized to 0.
    """

    loss: str
    n_features: int
    w: np.ndarray
    b: float

    def __init__(self, n_features: int, loss: str = "MSE"):
        """
        Parameters
        ----------
        n_features : int
            Number of features in the inputs the model will see. Determines
            the size of the weight vector w.
        loss : str, default "MSE"
            One of "MSE", "RMSE", or "MAE".
        """
        if loss not in LOSS_FUNCTIONS:
            raise ValueError("Loss must be one of the following:", LOSS_FUNCTIONS)
        if n_features < 1:
            raise ValueError("n_features must be >= 1")

        self.loss = loss
        self.n_features = n_features
        self.w = np.ones(n_features, dtype=float)
        self.b = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the model to a batch of inputs.

        For each row x_i of X, the prediction is

            y_hat_i = x_i . w + b

        Stacked over N samples this is just the matrix product

            y_hat = X @ w + b

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
            Predicted values.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (N, {self.n_features}); got {X.shape}")
        return X @ self.w + self.b

    def train_one_step(self, alpha: float, X: np.ndarray, y: np.ndarray) -> float:
        """
        Perform one gradient-descent step.

        Computes predictions, evaluates the chosen loss to get the gradients
        (delta_w, delta_b), and then updates the parameters with the standard
        gradient-descent rule:

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
            Target values.

        Returns
        -------
        float
            The loss value evaluated before the parameter update.
        """
        preds = self.predict(X)

        if self.loss == "MSE":
            error, delta_w, delta_b = loss.mse(X, y, preds)
        elif self.loss == "RMSE":
            error, delta_w, delta_b = loss.rmse(X, y, preds)
        elif self.loss == "MAE":
            error, delta_w, delta_b = loss.mae(X, y, preds)

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
            Target values.
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

        print("Training completed:")
        print("\tIterations:", num_iterations)
        print("\tFinal Error:", error)
        print(f"\tFinal Weights (w): {self.w}")
        print(f"\tFinal Bias    (b): {self.b}")
