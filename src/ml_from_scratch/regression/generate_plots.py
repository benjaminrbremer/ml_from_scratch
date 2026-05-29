"""
Generate the four figures used by linear_regression.md.

Run from anywhere:
    python src/ml_from_scratch/regression/generate_plots.py

This produces deterministic PNGs in `images/` next to this script. The plots
use the closed-form least-squares solution (normal equation) rather than
gradient descent so the figures are reproducible and do not depend on
learning-rate / convergence hyperparameters.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
IMAGES_DIR = HERE / "images"
IMAGES_DIR.mkdir(exist_ok=True)

SEED = 7


def true_function(x: np.ndarray) -> np.ndarray:
    """The hidden function the model is trying to recover."""
    return np.sin(1.5 * np.pi * x)


def make_dataset(n: int, noise_std: float, rng: np.random.Generator):
    x = rng.uniform(0.0, 1.0, size=n)
    y = true_function(x) + rng.normal(0.0, noise_std, size=n)
    return x, y


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Build Vandermonde-style feature matrix [1, x, x^2, ..., x^degree]."""
    return np.vander(x, N=degree + 1, increasing=True)


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form least squares: w = (X^T X)^+ X^T y."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge regression: w = (X^T X + lambda I)^{-1} X^T y.

    The bias column (first column of X) is left unpenalized -- standard practice.
    """
    n_features = X.shape[1]
    reg = lam * np.eye(n_features)
    reg[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# ---------------------------------------------------------------------------
# Fig 1: underfit / good fit / overfit on the same dataset
# ---------------------------------------------------------------------------
def fig1_polynomial_fits():
    rng = np.random.default_rng(SEED)
    x_train, y_train = make_dataset(n=20, noise_std=0.25, rng=rng)
    x_dense = np.linspace(0.0, 1.0, 400)
    y_true_dense = true_function(x_dense)

    degrees = [
        (1, "Underfit (degree 1)"),
        (3, "Good fit (degree 3)"),
        (15, "Overfit (degree 15)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    for ax, (deg, title) in zip(axes, degrees):
        X_train = polynomial_features(x_train, deg)
        w = fit_ols(X_train, y_train)
        X_dense = polynomial_features(x_dense, deg)
        y_pred_dense = X_dense @ w

        ax.scatter(
            x_train, y_train, color="black", s=30, zorder=3, label="Training data"
        )
        ax.plot(x_dense, y_true_dense, "--", color="gray", label="True function")
        ax.plot(x_dense, y_pred_dense, color="C0", linewidth=2, label="Model")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylim(-2.0, 2.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_ylabel("y")
    fig.suptitle("Bias-variance trade-off: polynomial fits of varying degree", y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig1_polynomial_fits.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: train vs validation error as a function of polynomial degree
# ---------------------------------------------------------------------------
def fig2_complexity_curve():
    rng = np.random.default_rng(SEED)
    x_train, y_train = make_dataset(n=30, noise_std=0.25, rng=rng)
    x_val, y_val = make_dataset(n=200, noise_std=0.25, rng=rng)

    degrees = list(range(1, 16))
    train_err, val_err = [], []
    for deg in degrees:
        X_tr = polynomial_features(x_train, deg)
        X_va = polynomial_features(x_val, deg)
        w = fit_ols(X_tr, y_train)
        train_err.append(mse(y_train, X_tr @ w))
        val_err.append(mse(y_val, X_va @ w))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(degrees, train_err, "o-", label="Training MSE", color="C0")
    ax.plot(degrees, val_err, "s-", label="Validation MSE", color="C3")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree (model complexity)")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("Train vs validation error as model complexity grows")
    ax.axvspan(2.5, 4.5, color="green", alpha=0.08, label="Sweet spot")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig2_complexity_curve.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: learning curves for a too-simple and a too-complex model
# ---------------------------------------------------------------------------
def fig3_learning_curves():
    rng = np.random.default_rng(SEED)
    x_val, y_val = make_dataset(n=500, noise_std=0.25, rng=rng)

    # Start at n=20 so the degree-15 system is overdetermined (n > degree + 1).
    # Below that the model interpolates exactly, training MSE collapses to ~0,
    # and the log-scale axis becomes useless.
    train_sizes = np.arange(20, 205, 10)
    results = {1: {"train": [], "val": []}, 15: {"train": [], "val": []}}

    for n in train_sizes:
        rng_inner = np.random.default_rng(SEED + int(n))
        x_tr, y_tr = make_dataset(n=int(n), noise_std=0.25, rng=rng_inner)
        for deg in (1, 15):
            X_tr = polynomial_features(x_tr, deg)
            X_va = polynomial_features(x_val, deg)
            w = fit_ols(X_tr, y_tr)
            results[deg]["train"].append(mse(y_tr, X_tr @ w))
            results[deg]["val"].append(mse(y_val, X_va @ w))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, deg, title in [
        (axes[0], 1, "Underfit model (degree 1) -- high bias"),
        (axes[1], 15, "Flexible model (degree 15) -- high variance"),
    ]:
        ax.plot(
            train_sizes, results[deg]["train"], "o-", label="Training MSE", color="C0"
        )
        ax.plot(
            train_sizes, results[deg]["val"], "s-", label="Validation MSE", color="C3"
        )
        ax.set_title(title)
        ax.set_xlabel("Number of training samples")
        ax.set_ylabel("MSE")
        ax.set_ylim(0, 0.35)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Learning curves: how training-set size affects each regime", y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig3_learning_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4: effect of L2 regularization strength on a degree-15 fit
# ---------------------------------------------------------------------------
def fig4_l2_regularization():
    rng = np.random.default_rng(SEED)
    x_train, y_train = make_dataset(n=20, noise_std=0.25, rng=rng)
    x_dense = np.linspace(0.0, 1.0, 400)
    y_true_dense = true_function(x_dense)

    degree = 15
    X_train = polynomial_features(x_train, degree)
    X_dense = polynomial_features(x_dense, degree)

    lambdas = [0.0, 1e-4, 1e-2, 1.0]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.scatter(x_train, y_train, color="black", s=30, zorder=3, label="Training data")
    ax.plot(x_dense, y_true_dense, "--", color="gray", label="True function")
    colors = ["C3", "C1", "C2", "C0"]
    for lam, color in zip(lambdas, colors):
        w = fit_ridge(X_train, y_train, lam)
        y_pred = X_dense @ w
        label = f"$\\lambda = {lam}$" if lam > 0 else "$\\lambda = 0$ (OLS)"
        ax.plot(x_dense, y_pred, color=color, linewidth=2, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-2.0, 2.0)
    ax.set_title("L2 regularization tames a degree-15 fit as $\\lambda$ grows")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig4_l2_regularization.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig1_polynomial_fits()
    fig2_complexity_curve()
    fig3_learning_curves()
    fig4_l2_regularization()
    print(f"Wrote 4 figures to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
