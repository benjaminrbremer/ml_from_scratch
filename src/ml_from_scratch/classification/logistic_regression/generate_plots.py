"""
Generate the figures used by logistic_regression.md.

Run from anywhere:
    python src/ml_from_scratch/classification/generate_plots.py

This produces deterministic PNGs in `images/` next to this script. The plots
fit logistic regression with L2-regularized Newton/IRLS (iteratively reweighted
least squares) rather than gradient descent, so the figures are reproducible
and do not depend on learning-rate / convergence hyperparameters. IRLS is the
classical analog of the normal equation for logistic regression: each
iteration solves a weighted least-squares problem, and it converges in
a handful of steps on these toy datasets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
IMAGES_DIR = HERE / "images"
IMAGES_DIR.mkdir(exist_ok=True)

SEED = 7


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable elementwise sigmoid (see logistic_regression._sigmoid)."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg])
    out[neg] = e / (1.0 + e)
    return out


def make_blobs(
    n: int, rng: np.random.Generator, noise: float = 0.30
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two interleaved-moons-style classes in 2D.

    Generates a non-linearly-separable dataset so that fixed-complexity logistic
    regression underfits but a polynomial-feature version can fit cleanly.
    Higher `noise` makes the moons overlap more, which is what makes overfitting
    visually obvious (a high-degree model contorts to chase every flipped point).
    """
    n0 = n // 2
    n1 = n - n0
    # "Moon" 0: a downward arc centered around the origin.
    t0 = rng.uniform(0.0, np.pi, size=n0)
    x0 = np.cos(t0) + rng.normal(0.0, noise, size=n0)
    y0 = np.sin(t0) + rng.normal(0.0, noise, size=n0)
    # "Moon" 1: an upward arc, shifted.
    t1 = rng.uniform(0.0, np.pi, size=n1)
    x1 = 1.0 - np.cos(t1) + rng.normal(0.0, noise, size=n1)
    y1 = 0.5 - np.sin(t1) + rng.normal(0.0, noise, size=n1)

    X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
    y = np.concatenate([np.zeros(n0), np.ones(n1)])

    # Shuffle so train/val splits aren't class-stratified by index.
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def polynomial_features_2d(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Build all 2D monomials up to total degree `degree`, *including* the bias
    column at index 0 ("1"). The resulting design matrix has
    (degree + 1) * (degree + 2) / 2 columns.

    For degree 2 with inputs (x1, x2): columns are [1, x1, x2, x1^2, x1 x2, x2^2].
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    cols = []
    for d in range(degree + 1):
        for k in range(d + 1):
            cols.append((x1 ** (d - k)) * (x2**k))
    return np.column_stack(cols)


def fit_irls(
    Phi: np.ndarray, y: np.ndarray, lam: float = 1e-6, max_iter: int = 50
) -> np.ndarray:
    """
    Fit logistic regression by L2-regularized iteratively reweighted least
    squares. `Phi` already contains the bias column at index 0, which is left
    unpenalized.

    Newton update (per iteration):
        z       = Phi @ theta
        p       = sigmoid(z)
        W       = diag(p * (1 - p))                       (N x N, stored as vector)
        H       = Phi.T @ W @ Phi + lam * R               (R zeroes out bias entry)
        g       = Phi.T @ (p - y) + lam * R @ theta
        theta  -= solve(H, g)

    Returns theta of shape (Phi.shape[1],). Stops early when the change in
    theta drops below 1e-9 in infinity norm.
    """
    n_features = Phi.shape[1]
    R = np.eye(n_features)
    R[0, 0] = 0.0  # don't penalize bias

    theta = np.zeros(n_features)
    for _ in range(max_iter):
        z = Phi @ theta
        p = sigmoid(z)
        # Clip W away from 0 for numerical conditioning.
        W = np.clip(p * (1.0 - p), 1e-9, None)
        WP = Phi * W[:, None]
        H = Phi.T @ WP + lam * R
        g = Phi.T @ (p - y) + lam * (R @ theta)
        step = np.linalg.solve(H, g)
        theta -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return theta


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def decision_grid(
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    degree: int,
    theta: np.ndarray,
    n: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(*x_lim, n)
    ys = np.linspace(*y_lim, n)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    Phi_grid = polynomial_features_2d(grid, degree)
    P = sigmoid(Phi_grid @ theta).reshape(XX.shape)
    return XX, YY, P


# ---------------------------------------------------------------------------
# Fig 0: the sigmoid curve itself
# ---------------------------------------------------------------------------
def fig0_sigmoid():
    z = np.linspace(-7.0, 7.0, 400)
    p = sigmoid(z)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(z, p, color="C0", linewidth=2.2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("z = X w + b  (logit / raw score)")
    ax.set_ylabel(r"$\sigma(z) = 1 / (1 + e^{-z})$")
    ax.set_title("The sigmoid function maps any real number to (0, 1)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig0_sigmoid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1: underfit / good fit / overfit decision boundaries
# ---------------------------------------------------------------------------
def fig1_decision_boundaries():
    rng = np.random.default_rng(SEED)
    # Smaller, noisier dataset so that high-degree models visibly chase
    # individual mislabeled-looking points.
    X, y = make_blobs(n=40, rng=rng, noise=0.35)
    x_lim = (X[:, 0].min() - 0.4, X[:, 0].max() + 0.4)
    y_lim = (X[:, 1].min() - 0.4, X[:, 1].max() + 0.4)

    degrees = [
        (1, "Underfit (degree 1)", 1e-4),
        (3, "Good fit (degree 3)", 1e-4),
        (10, "Overfit (degree 10)", 1e-8),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), sharey=True)
    for ax, (deg, title, lam) in zip(axes, degrees):
        Phi = polynomial_features_2d(X, deg)
        theta = fit_irls(Phi, y, lam=lam)
        XX, YY, P = decision_grid(x_lim, y_lim, deg, theta)

        ax.contourf(
            XX,
            YY,
            P,
            levels=20,
            cmap="RdBu_r",
            alpha=0.55,
            vmin=0.0,
            vmax=1.0,
        )
        ax.contour(XX, YY, P, levels=[0.5], colors="black", linewidths=1.5)
        ax.scatter(
            X[y == 0, 0], X[y == 0, 1], c="C0", edgecolor="black", s=35, label="class 0"
        )
        ax.scatter(
            X[y == 1, 0], X[y == 1, 1], c="C3", edgecolor="black", s=35, label="class 1"
        )
        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("$x_2$")
    fig.suptitle(
        "Decision boundaries at increasing model complexity (polynomial features)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        IMAGES_DIR / "fig1_decision_boundaries.png", dpi=130, bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: train vs validation log loss as a function of polynomial degree
# ---------------------------------------------------------------------------
def fig2_complexity_curve():
    rng = np.random.default_rng(SEED)
    # Small noisy training set so high-degree models can chase noise; larger
    # validation set so the val curve is smooth.
    X_train, y_train = make_blobs(n=60, rng=rng, noise=0.35)
    X_val, y_val = make_blobs(n=600, rng=rng, noise=0.35)

    # Fixed tiny ridge across all degrees to keep IRLS numerically stable
    # without meaningfully constraining the fit. We stop at degree 8 because
    # at higher degrees with 60 samples the design matrix becomes severely
    # ill-conditioned (more features than samples) and the IRLS solution
    # is dominated by numerical noise rather than overfitting signal.
    LAM = 1e-3
    degrees = list(range(1, 9))
    train_err, val_err = [], []
    for deg in degrees:
        Phi_tr = polynomial_features_2d(X_train, deg)
        Phi_va = polynomial_features_2d(X_val, deg)
        theta = fit_irls(Phi_tr, y_train, lam=LAM)
        train_err.append(log_loss(y_train, sigmoid(Phi_tr @ theta)))
        val_err.append(log_loss(y_val, sigmoid(Phi_va @ theta)))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(degrees, train_err, "o-", label="Training log loss", color="C0")
    ax.plot(degrees, val_err, "s-", label="Validation log loss", color="C3")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree (model complexity)")
    ax.set_ylabel("Log loss (log scale)")
    ax.set_title("Train vs validation log loss as model complexity grows")
    ax.axvspan(1.5, 3.5, color="green", alpha=0.08, label="Sweet spot")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig2_complexity_curve.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: learning curves for a too-simple and a too-complex model
# ---------------------------------------------------------------------------
def fig3_learning_curves():
    rng_val = np.random.default_rng(SEED)
    X_val, y_val = make_blobs(n=600, rng=rng_val, noise=0.35)

    # Average over several seeds at each training size to smooth out the
    # sampling noise inherent in learning curves on small datasets.
    train_sizes = np.arange(40, 305, 20)
    n_seeds = 8
    flex_degree = 6
    results = {1: {"train": [], "val": []}, flex_degree: {"train": [], "val": []}}

    for n in train_sizes:
        for deg in (1, flex_degree):
            tr_runs, va_runs = [], []
            # Slightly stronger ridge for the high-degree case so IRLS does
            # not occasionally diverge on adversarial-looking small samples;
            # we want a clean learning curve, not a fight with conditioning.
            lam = 1e-3 if deg == 1 else 1e-2
            for s in range(n_seeds):
                rng_inner = np.random.default_rng(SEED + int(n) * 7 + s)
                X_tr, y_tr = make_blobs(n=int(n), rng=rng_inner, noise=0.35)
                Phi_tr = polynomial_features_2d(X_tr, deg)
                Phi_va = polynomial_features_2d(X_val, deg)
                theta = fit_irls(Phi_tr, y_tr, lam=lam)
                tr_runs.append(log_loss(y_tr, sigmoid(Phi_tr @ theta)))
                va_runs.append(log_loss(y_val, sigmoid(Phi_va @ theta)))
            # Median is robust to occasional IRLS-divergence outliers across seeds.
            results[deg]["train"].append(float(np.median(tr_runs)))
            results[deg]["val"].append(float(np.median(va_runs)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, deg, title in [
        (axes[0], 1, "Underfit model (degree 1) -- high bias"),
        (
            axes[1],
            flex_degree,
            f"Flexible model (degree {flex_degree}) -- high variance",
        ),
    ]:
        ax.plot(
            train_sizes,
            results[deg]["train"],
            "o-",
            label="Training log loss",
            color="C0",
        )
        ax.plot(
            train_sizes,
            results[deg]["val"],
            "s-",
            label="Validation log loss",
            color="C3",
        )
        ax.set_title(title)
        ax.set_xlabel("Number of training samples")
        ax.set_ylabel("Log loss")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Learning curves: how training-set size affects each regime", y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig3_learning_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4: effect of L2 regularization strength on a degree-10 decision boundary
# ---------------------------------------------------------------------------
def fig4_l2_regularization():
    rng = np.random.default_rng(SEED)
    # Same small-noisy dataset as fig1 so that the lam=very-small case
    # produces the visibly wiggly overfit boundary we want to contrast against.
    X, y = make_blobs(n=40, rng=rng, noise=0.35)
    x_lim = (X[:, 0].min() - 0.4, X[:, 0].max() + 0.4)
    y_lim = (X[:, 1].min() - 0.4, X[:, 1].max() + 0.4)

    degree = 10
    lambdas = [1e-8, 1e-2, 1.0, 10.0]
    titles = [
        r"$\lambda = 10^{-8}$ (effectively unregularized)",
        r"$\lambda = 10^{-2}$",
        r"$\lambda = 1$",
        r"$\lambda = 10$ (heavily regularized)",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    Phi = polynomial_features_2d(X, degree)
    for ax, lam, title in zip(axes, lambdas, titles):
        theta = fit_irls(Phi, y, lam=lam)
        XX, YY, P = decision_grid(x_lim, y_lim, degree, theta)
        ax.contourf(XX, YY, P, levels=20, cmap="RdBu_r", alpha=0.55, vmin=0.0, vmax=1.0)
        ax.contour(XX, YY, P, levels=[0.5], colors="black", linewidths=1.5)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="C0", edgecolor="black", s=30)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="C3", edgecolor="black", s=30)
        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("$x_2$")
    fig.suptitle(
        "L2 regularization smooths the decision boundary as $\\lambda$ grows",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig4_l2_regularization.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig0_sigmoid()
    fig1_decision_boundaries()
    fig2_complexity_curve()
    fig3_learning_curves()
    fig4_l2_regularization()
    print(f"Wrote 5 figures to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
