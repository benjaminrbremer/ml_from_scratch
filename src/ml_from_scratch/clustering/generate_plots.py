"""
Generate figures for k_means.md.

Three plots are produced:

    images/convergence_trace.png
        Four panels showing the cluster assignments and centroid positions
        at iterations 1, 2, 3, and convergence on a 3-cluster 2D dataset.
        Illustrates how k-means refines assignments and moves centroids.

    images/elbow_curve.png
        Inertia vs. K for K=1..8. The synthetic dataset has 3 true clusters,
        so the elbow appears at K=3.

    images/init_comparison.png
        Side-by-side final clustering from random initialization vs.
        K-means++ on a dataset where random init is likely to collapse.
        Uses a fixed seed to show a bad-luck random run and a good kmeans++.

Run with:
    python src/ml_from_scratch/clustering/generate_plots.py
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from ml_from_scratch.clustering.k_means import k_means

IMAGES_DIR = pathlib.Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
CENTROID_STYLE = dict(s=200, marker="X", edgecolors="black", linewidths=1.2, zorder=5)


def _make_3cluster_data(seed: int = 0) -> np.ndarray:
    """Three well-separated Gaussian blobs in 2D."""
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            rng.normal([0.0, 0.0], 0.6, (80, 2)),
            rng.normal([4.0, 0.0], 0.6, (80, 2)),
            rng.normal([2.0, 3.5], 0.6, (80, 2)),
        ]
    )


# ---------------------------------------------------------------------------
# Figure 1: convergence trace
# ---------------------------------------------------------------------------


def plot_convergence_trace():
    X = _make_3cluster_data(seed=1)
    K = 3
    SNAPSHOTS = [1, 2, 3, None]  # None = run to convergence

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
    fig.suptitle("K-means convergence trace (K=3)", fontsize=13)

    rng = np.random.default_rng(7)
    model = k_means(k=K, n_features=2, init="kmeans++", random_state=7)
    model.centroids = model._init_centroids(X, rng)

    iteration = 0
    for ax, target in zip(axes, SNAPSHOTS):
        if target is None:
            # Run to convergence
            for _ in range(300):
                _, done = model.train_one_step(X, tol=1e-4)
                iteration += 1
                if done:
                    break
            title = f"Converged (iter {iteration})"
        else:
            while iteration < target:
                model.train_one_step(X, tol=1e-5)
                iteration += 1
            title = f"After iteration {iteration}"

        labels = model.predict(X)
        for k in range(K):
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], c=PALETTE[k], s=20, alpha=0.6)
        ax.scatter(
            model.centroids[:, 0],
            model.centroids[:, 1],
            c=PALETTE[:K],
            **CENTROID_STYLE,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    out = IMAGES_DIR / "convergence_trace.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: elbow curve
# ---------------------------------------------------------------------------


def plot_elbow_curve():
    X = _make_3cluster_data(seed=2)
    ks = range(1, 9)
    inertias = []
    for k in ks:
        m = k_means(k=k, n_features=2, init="kmeans++", random_state=42)
        m.fit(X, max_iterations=300)
        inertias.append(m.inertia(X))

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(list(ks), inertias, "o-", color="#377eb8", linewidth=2, markersize=7)
    ax.axvline(3, color="#e41a1c", linestyle="--", linewidth=1.4, label="K=3 (true)")
    ax.set_xlabel("K (number of clusters)", fontsize=12)
    ax.set_ylabel("Inertia (WCSS)", fontsize=12)
    ax.set_title("Elbow curve — inertia vs. K", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks(list(ks))

    out = IMAGES_DIR / "elbow_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: initialization comparison
# ---------------------------------------------------------------------------


def plot_init_comparison():
    """
    Three balanced, well-separated clusters. With random_state=0, random init
    picks two centroids from the same cluster (indices from cluster 1); the
    two real clusters merge into one and the result is a bad local minimum
    (inertia ~2767). K-means++ with the same seed spreads the centroids and
    finds the correct solution (inertia ~230).
    """
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal([0.0, 0.0], 0.8, (60, 2)),
            rng.normal([10.0, 0.0], 0.8, (60, 2)),
            rng.normal([5.0, 8.0], 0.8, (60, 2)),
        ]
    )

    K = 3
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    fig.suptitle("Random init vs. K-means++ — same seed, same data", fontsize=13)

    configs = [
        ("random", 0, "Random init (seed=0) — bad local min"),
        ("kmeans++", 0, "K-means++ (seed=0) — correct solution"),
    ]

    for ax, (init, seed, title) in zip(axes, configs):
        m = k_means(k=K, n_features=2, init=init, random_state=seed)
        m.fit(X, max_iterations=300)
        labels = m.predict(X)
        for k in range(K):
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], c=PALETTE[k], s=15, alpha=0.5)
        ax.scatter(
            m.centroids[:, 0],
            m.centroids[:, 1],
            c=PALETTE[:K],
            **CENTROID_STYLE,
        )
        ax.set_title(f"{title}\nInertia = {m.inertia(X):.1f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    out = IMAGES_DIR / "init_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_convergence_trace()
    plot_elbow_curve()
    plot_init_comparison()
    print("Done.")
