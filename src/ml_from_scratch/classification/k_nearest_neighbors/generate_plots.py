"""
Generate the figures used by k_nearest_neighbors.md.

Run from anywhere:
    python src/ml_from_scratch/classification/k_nearest_neighbors/generate_plots.py

This produces deterministic PNGs in `images/` next to this script. Unlike
the linear / logistic regression figure scripts, this one uses the KNN
class itself rather than a closed-form analog: brute-force KNN *is* the
algorithm, with no hyperparameters to converge -- given the same fit
data and the same k, the prediction is exact and reproducible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ml_from_scratch.classification.k_nearest_neighbors.k_nearest_neighbors import (
    k_nearest_neighbors,
)

HERE = Path(__file__).resolve().parent
IMAGES_DIR = HERE / "images"
IMAGES_DIR.mkdir(exist_ok=True)

SEED = 7


def make_blobs(
    n: int, rng: np.random.Generator, noise: float = 0.30
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two interleaved-moons-style classes in 2D.

    Same data generator as classification/generate_plots.py (logistic
    regression). Reproduced here so this script is self-contained, per
    the existing "figure scripts are self-contained" convention.
    """
    n0 = n // 2
    n1 = n - n0
    t0 = rng.uniform(0.0, np.pi, size=n0)
    x0 = np.cos(t0) + rng.normal(0.0, noise, size=n0)
    y0 = np.sin(t0) + rng.normal(0.0, noise, size=n0)
    t1 = rng.uniform(0.0, np.pi, size=n1)
    x1 = 1.0 - np.cos(t1) + rng.normal(0.0, noise, size=n1)
    y1 = 0.5 - np.sin(t1) + rng.normal(0.0, noise, size=n1)

    X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
    y = np.concatenate([np.zeros(n0), np.ones(n1)])

    idx = rng.permutation(len(y))
    return X[idx], y[idx].astype(int)


def decision_grid(
    clf: k_nearest_neighbors,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    n: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate `clf.predict` on a dense grid spanning x_lim x y_lim so we
    can contour the decision surface. n=200 is a good balance between
    smoothness and runtime; KNN's prediction cost scales linearly with
    grid size and training-set size.
    """
    xs = np.linspace(*x_lim, n)
    ys = np.linspace(*y_lim, n)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    P = clf.predict(grid).reshape(XX.shape)
    return XX, YY, P


# ---------------------------------------------------------------------------
# Fig 1: underfit / good fit / overfit decision boundaries via k
# ---------------------------------------------------------------------------
def fig1_decision_boundaries():
    rng = np.random.default_rng(SEED)
    # Noisy two-moons -- enough overlap that small k will visibly chase
    # individual mislabeled-looking points.
    X, y = make_blobs(n=120, rng=rng, noise=0.35)
    x_lim = (X[:, 0].min() - 0.4, X[:, 0].max() + 0.4)
    y_lim = (X[:, 1].min() - 0.4, X[:, 1].max() + 0.4)

    configs = [
        (1, "Overfit (k = 1) -- jagged"),
        (15, "Good fit (k = 15)"),
        (101, "Underfit (k = 101) -- nearly flat"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), sharey=True)
    for ax, (k, title) in zip(axes, configs):
        clf = k_nearest_neighbors(k=k)
        clf.fit(X, y)
        XX, YY, P = decision_grid(clf, x_lim, y_lim)

        ax.contourf(XX, YY, P, levels=20, cmap="RdBu_r", alpha=0.55, vmin=0.0, vmax=1.0)
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
    fig.suptitle("KNN decision boundaries at increasing k (model smoothness)", y=1.02)
    fig.tight_layout()
    fig.savefig(
        IMAGES_DIR / "fig1_decision_boundaries.png", dpi=130, bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: train vs validation 0-1 error as a function of k
# ---------------------------------------------------------------------------
def fig2_complexity_curve():
    rng = np.random.default_rng(SEED)
    # Train set small enough that k=1 visibly memorizes; val set large
    # enough that the val curve is smooth.
    X_train, y_train = make_blobs(n=120, rng=rng, noise=0.35)
    X_val, y_val = make_blobs(n=600, rng=rng, noise=0.35)

    # Sweep k. The x-axis convention in KNN is k itself, but model
    # complexity *decreases* with k (k=1 = max complexity, k=N = constant).
    # We log-space k from 1 to ~N_train so the small-k regime is visible.
    ks = np.unique(np.round(np.geomspace(1, len(X_train) - 1, 25)).astype(int))
    train_err, val_err = [], []
    for k in ks:
        clf = k_nearest_neighbors(k=int(k))
        clf.fit(X_train, y_train)
        train_err.append(float(np.mean(clf.predict_class(X_train) != y_train)))
        val_err.append(float(np.mean(clf.predict_class(X_val) != y_val)))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ks, train_err, "o-", label="Training error", color="C0")
    ax.plot(ks, val_err, "s-", label="Validation error", color="C3")
    ax.set_xscale("log")
    ax.set_xlabel("k  (number of neighbors)  --  model complexity decreases as k grows")
    ax.set_ylabel("0-1 error rate")
    ax.set_title("Train vs validation error as k grows (the bias-variance dial)")
    # Annotate the regimes verbally instead of by axis flip to keep the
    # standard "k on the x-axis" convention.
    ax.text(
        0.02,
        0.98,
        "k small\n= high variance\n(overfit)",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    ax.text(
        0.98,
        0.98,
        "k large\n= high bias\n(underfit)",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig2_complexity_curve.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: learning curves for a too-flexible and a too-smooth k
# ---------------------------------------------------------------------------
def fig3_learning_curves():
    rng_val = np.random.default_rng(SEED)
    X_val, y_val = make_blobs(n=600, rng=rng_val, noise=0.35)

    # Median over a handful of seeds at each training size, like the
    # logistic-regression learning-curve plot does.
    train_sizes = np.arange(20, 305, 20)
    n_seeds = 6
    small_k = 1
    large_k = 51
    results = {
        small_k: {"train": [], "val": []},
        large_k: {"train": [], "val": []},
    }

    for n in train_sizes:
        for k in (small_k, large_k):
            # Skip configurations where N < k; KNN can't fit.
            if n < k:
                results[k]["train"].append(np.nan)
                results[k]["val"].append(np.nan)
                continue
            tr_runs, va_runs = [], []
            for s in range(n_seeds):
                rng_inner = np.random.default_rng(SEED + int(n) * 7 + s)
                X_tr, y_tr = make_blobs(n=int(n), rng=rng_inner, noise=0.35)
                clf = k_nearest_neighbors(k=k)
                clf.fit(X_tr, y_tr)
                tr_runs.append(float(np.mean(clf.predict_class(X_tr) != y_tr)))
                va_runs.append(float(np.mean(clf.predict_class(X_val) != y_val)))
            results[k]["train"].append(float(np.median(tr_runs)))
            results[k]["val"].append(float(np.median(va_runs)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, k, title in [
        (axes[0], small_k, f"Flexible (k = {small_k}) -- high variance"),
        (axes[1], large_k, f"Smooth (k = {large_k}) -- high bias"),
    ]:
        ax.plot(
            train_sizes, results[k]["train"], "o-", label="Training error", color="C0"
        )
        ax.plot(
            train_sizes, results[k]["val"], "s-", label="Validation error", color="C3"
        )
        ax.set_title(title)
        ax.set_xlabel("Number of training samples")
        ax.set_ylabel("0-1 error rate")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Learning curves: how training-set size affects each k regime", y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig3_learning_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4: the effect of feature scaling on the decision boundary
# ---------------------------------------------------------------------------
def fig4_feature_scaling():
    rng = np.random.default_rng(SEED)
    X, y = make_blobs(n=200, rng=rng, noise=0.30)

    # Construct a deliberately bad scaling: inflate feature 0 by 100x.
    # Euclidean distance is now dominated by feature 0; the boundary
    # should become a near-vertical strip ignoring feature 1.
    X_bad = X.copy()
    X_bad[:, 0] = X_bad[:, 0] * 100.0

    # Standardize (zero mean, unit variance per feature) to restore the
    # intended geometry.
    means = X_bad.mean(axis=0)
    stds = X_bad.std(axis=0)
    X_scaled = (X_bad - means) / stds

    k = 15
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, X_train, title in [
        (axes[0], X_bad, "Unscaled: feature 0 is 100x feature 1"),
        (axes[1], X_scaled, "Standardized: zero mean, unit variance"),
    ]:
        clf = k_nearest_neighbors(k=k)
        clf.fit(X_train, y)
        x_lim = (
            X_train[:, 0].min() - 0.2 * np.ptp(X_train[:, 0]),
            X_train[:, 0].max() + 0.2 * np.ptp(X_train[:, 0]),
        )
        y_lim = (
            X_train[:, 1].min() - 0.2 * np.ptp(X_train[:, 1]),
            X_train[:, 1].max() + 0.2 * np.ptp(X_train[:, 1]),
        )
        XX, YY, P = decision_grid(clf, x_lim, y_lim, n=200)
        ax.contourf(XX, YY, P, levels=20, cmap="RdBu_r", alpha=0.55, vmin=0.0, vmax=1.0)
        ax.contour(XX, YY, P, levels=[0.5], colors="black", linewidths=1.5)
        ax.scatter(
            X_train[y == 0, 0],
            X_train[y == 0, 1],
            c="C0",
            edgecolor="black",
            s=25,
            label="class 0",
        )
        ax.scatter(
            X_train[y == 1, 0],
            X_train[y == 1, 1],
            c="C3",
            edgecolor="black",
            s=25,
            label="class 1",
        )
        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"KNN (k = {k}) is dominated by the largest-scale feature unless inputs are standardized",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig4_feature_scaling.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig1_decision_boundaries()
    fig2_complexity_curve()
    fig3_learning_curves()
    fig4_feature_scaling()
    print(f"Wrote 4 figures to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
