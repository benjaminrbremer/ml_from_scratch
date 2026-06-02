"""
Generate the figures used by decision_tree.md.

Run from anywhere:
    python src/ml_from_scratch/classification/trees/generate_plots.py

Produces deterministic PNGs in `images/` next to this script. Every plot uses
only the decision_tree and impurity modules from this package — no sklearn.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ml_from_scratch.classification.trees.decision_tree import decision_tree
from ml_from_scratch.classification.trees.impurity import (
    entropy,
    gini,
    information_gain,
)

HERE = Path(__file__).resolve().parent
IMAGES_DIR = HERE / "images"
IMAGES_DIR.mkdir(exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

# ── colour palette ────────────────────────────────────────────────────────────
C0 = "#2196F3"  # blue  — class 0 / primary line
C1 = "#F44336"  # red   — class 1 / secondary line
CBOUND = "#444444"  # dark grey — decision boundary
CGRID = "#f5f5f5"  # very light grey — background fill


# ── helpers ───────────────────────────────────────────────────────────────────


def make_dataset(n: int = 300, noise: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """
    Two interleaved half-moons — a canonical non-linearly separable dataset.
    Reproducible via module-level SEED.
    """
    t0 = RNG.uniform(0, np.pi, n // 2)
    t1 = RNG.uniform(0, np.pi, n // 2)

    X0 = np.column_stack([np.cos(t0), np.sin(t0)]) + RNG.normal(0, noise, (n // 2, 2))
    X1 = np.column_stack([1 - np.cos(t1), 1 - np.sin(t1) - 0.5]) + RNG.normal(
        0, noise, (n // 2, 2)
    )
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
    perm = RNG.permutation(len(y))
    return X[perm], y[perm]


def plot_boundary(ax, model, X, y, title: str, resolution: int = 300) -> None:
    """Draw scatter + filled decision-region background on ax."""
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=[C0, C1], alpha=0.15)
    ax.contour(xx, yy, Z, levels=[0.5], colors=[CBOUND], linewidths=1.2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c=C0, s=18, alpha=0.7, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=C1, s=18, alpha=0.7, label="class 1")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


# ── fig 0 : impurity curves ───────────────────────────────────────────────────


def fig0_impurity_curves() -> None:
    """
    Entropy and Gini impurity as a function of the class-1 probability p.
    Both measures peak at p = 0.5 (maximum uncertainty) and are zero at
    p = 0 or p = 1 (pure nodes).
    """
    p = np.linspace(0, 1, 500)
    # avoid log(0)
    eps = 1e-12
    p_safe = np.clip(p, eps, 1 - eps)
    ent = -(p_safe * np.log2(p_safe) + (1 - p_safe) * np.log2(1 - p_safe))
    gin = 2 * p * (1 - p)  # = 1 - p^2 - (1-p)^2, same as gini formula

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(p, ent, color=C0, lw=2, label="Entropy (bits)")
    ax.plot(p, gin, color=C1, lw=2, linestyle="--", label="Gini impurity")

    ax.axvline(0.5, color="grey", lw=0.8, linestyle=":")
    ax.set_xlabel("$p$ (fraction of class 1)", fontsize=12)
    ax.set_ylabel("Impurity", fontsize=12)
    ax.set_title("Impurity measures vs. class balance", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig0_impurity_curves.png", dpi=150)
    plt.close(fig)
    print("Saved fig0_impurity_curves.png")


# ── fig 1 : split selection ───────────────────────────────────────────────────


def fig1_split_selection() -> None:
    """
    Information gain as a function of the threshold for a single feature.
    Shows how the CART algorithm sweeps candidate thresholds to find the
    optimal split point.
    """
    # 1-D dataset: feature 0 drawn from two Gaussians
    rng_local = np.random.default_rng(SEED + 1)
    x0 = rng_local.normal(1.5, 0.5, 60)
    x1 = rng_local.normal(3.0, 0.6, 60)
    x = np.concatenate([x0, x1])
    y = np.array([0] * 60 + [1] * 60)

    # Sweep thresholds
    x_sorted = np.sort(np.unique(x))
    thresholds = (x_sorted[:-1] + x_sorted[1:]) / 2.0
    gains = []
    for t in thresholds:
        mask = x <= t
        gains.append(information_gain(y, y[mask], y[~mask]))
    gains = np.array(gains)

    best_idx = int(np.argmax(gains))
    best_t = thresholds[best_idx]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 5), gridspec_kw={"height_ratios": [1, 2]}
    )

    # Top: data distribution
    ax_top.scatter(x[y == 0], np.zeros(60), c=C0, s=20, alpha=0.6, label="class 0")
    ax_top.scatter(x[y == 1], np.zeros(60), c=C1, s=20, alpha=0.6, label="class 1")
    ax_top.axvline(best_t, color=CBOUND, lw=1.5, linestyle="--")
    ax_top.set_yticks([])
    ax_top.set_xlim(x.min() - 0.2, x.max() + 0.2)
    ax_top.set_title(
        "Feature values (top) and information gain vs. threshold (bottom)", fontsize=11
    )
    ax_top.legend(loc="upper right", fontsize=9)

    # Bottom: IG curve
    ax_bot.plot(thresholds, gains, color=C0, lw=1.5)
    ax_bot.scatter(
        [best_t],
        [gains[best_idx]],
        color=CBOUND,
        zorder=5,
        s=60,
        label=f"best threshold = {best_t:.2f}\nIG = {gains[best_idx]:.4f}",
    )
    ax_bot.axvline(best_t, color=CBOUND, lw=1.5, linestyle="--")
    ax_bot.set_xlabel("Threshold", fontsize=11)
    ax_bot.set_ylabel("Information Gain (bits)", fontsize=11)
    ax_bot.legend(fontsize=9)
    ax_bot.set_xlim(x.min() - 0.2, x.max() + 0.2)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig1_split_selection.png", dpi=150)
    plt.close(fig)
    print("Saved fig1_split_selection.png")


# ── fig 2 : decision boundaries at multiple depths ───────────────────────────


def fig2_decision_boundaries() -> None:
    """
    Decision boundaries produced by trees of different max_depth values on
    the half-moons dataset. Illustrates how increasing depth allows the tree
    to capture more complex (but potentially overfit) boundaries.
    """
    X, y = make_dataset(n=400)
    depths = [1, 3, 6]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, d in zip(axes, depths):
        dt = decision_tree(max_depth=d)
        dt.train(X, y)
        acc = float(np.mean(dt.predict_class(X) == y))
        plot_boundary(ax, dt, X, y, title=f"max_depth={d}  (train acc={acc:.2f})")

    axes[0].legend(loc="lower right", fontsize=8, markerscale=1.2)
    fig.suptitle("Decision boundaries at increasing tree depth", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(
        IMAGES_DIR / "fig2_decision_boundaries.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("Saved fig2_decision_boundaries.png")


# ── fig 3 : complexity curve ──────────────────────────────────────────────────


def fig3_complexity_curve() -> None:
    """
    Train and test accuracy as a function of max_depth. Illustrates the
    bias-variance trade-off: shallow trees underfit (high bias), deep trees
    overfit (high variance).
    """
    X, y = make_dataset(n=600)
    n_train = 400
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    depths = list(range(1, 20))
    train_accs, test_accs = [], []
    for d in depths:
        dt = decision_tree(max_depth=d)
        dt.train(X_train, y_train)
        train_accs.append(float(np.mean(dt.predict_class(X_train) == y_train)))
        test_accs.append(float(np.mean(dt.predict_class(X_test) == y_test)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        depths,
        train_accs,
        color=C0,
        lw=2,
        marker="o",
        markersize=5,
        label="Train accuracy",
    )
    ax.plot(
        depths,
        test_accs,
        color=C1,
        lw=2,
        marker="s",
        markersize=5,
        label="Test accuracy",
    )
    ax.set_xlabel("max_depth", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Complexity curve: depth vs. accuracy", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.02)

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig3_complexity_curve.png", dpi=150)
    plt.close(fig)
    print("Saved fig3_complexity_curve.png")


# ── fig 4 : learning curves ───────────────────────────────────────────────────


def fig4_learning_curves() -> None:
    """
    Train and test accuracy as a function of training-set size with a fixed
    max_depth=5. Illustrates that more data generally helps generalisation.
    """
    X_full, y_full = make_dataset(n=800)
    X_test, y_test = X_full[600:], y_full[600:]

    sizes = list(range(20, 601, 20))
    train_accs, test_accs = [], []
    for n in sizes:
        X_train, y_train = X_full[:n], y_full[:n]
        dt = decision_tree(max_depth=5)
        dt.train(X_train, y_train)
        train_accs.append(float(np.mean(dt.predict_class(X_train) == y_train)))
        test_accs.append(float(np.mean(dt.predict_class(X_test) == y_test)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, train_accs, color=C0, lw=2, label="Train accuracy")
    ax.plot(sizes, test_accs, color=C1, lw=2, label="Test accuracy")
    ax.set_xlabel("Training-set size", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Learning curves (max_depth=5)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.02)

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig4_learning_curves.png", dpi=150)
    plt.close(fig)
    print("Saved fig4_learning_curves.png")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig0_impurity_curves()
    fig1_split_selection()
    fig2_decision_boundaries()
    fig3_complexity_curve()
    fig4_learning_curves()
    print("All figures written to", IMAGES_DIR)
