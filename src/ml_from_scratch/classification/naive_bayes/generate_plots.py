"""
Generate the figures for naive_bayes.md.

Run from any directory:
    python src/ml_from_scratch/classification/naive_bayes/generate_plots.py

Requires matplotlib (install via: uv pip install -e ".[docs]")

Figures
-------
fig0_bayes_theorem.png      : 1D illustration of Bayes' theorem
fig1_class_conditionals.png : Learned Gaussians overlaid on histograms
fig2_decision_boundary.png  : Decision boundary vs. logistic regression
fig3_naive_assumption.png   : Accuracy vs. feature correlation strength
fig4_multiclass.png         : Three-class classification with Gaussian contours
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal, norm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMG_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(_IMG_DIR, exist_ok=True)


def _sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg])
    out[neg] = e / (1.0 + e)
    return out


def _fit_nb(X, y):
    """Return (means, variances, log_priors, classes) — minimal inline fit."""
    classes = np.unique(y)
    K, n = len(classes), X.shape[1]
    means = np.zeros((K, n))
    variances = np.zeros((K, n))
    log_priors = np.zeros(K)
    for k, cls in enumerate(classes):
        Xk = X[y == cls]
        means[k] = Xk.mean(0)
        variances[k] = Xk.var(0) + 1e-9 * Xk.var(0).max()
        log_priors[k] = np.log(len(Xk) / len(X))
    return means, variances, log_priors, classes


def _nb_log_posterior(X, means, variances, log_priors):
    """Return (N, K) unnormalized log-posteriors."""
    diff = X[:, np.newaxis, :] - means[np.newaxis, :, :]
    log_ll = -0.5 * (np.log(2 * np.pi * variances) + diff**2 / variances)
    return log_ll.sum(2) + log_priors


def _fit_logreg(X, y, alpha=0.1, n_iter=2000):
    """L2-regularized logistic regression via gradient descent."""
    w = np.zeros(X.shape[1])
    b = 0.0
    lam = 0.01
    for _ in range(n_iter):
        z = X @ w + b
        p = _sigmoid(z)
        r = p - y
        w -= alpha * ((X.T @ r) / len(y) + lam * w)
        b -= alpha * r.mean()
    return w, b


# ---------------------------------------------------------------------------
# fig0 — Bayes' theorem in 1D
# ---------------------------------------------------------------------------


def fig0_bayes_theorem():
    """
    Shows a simple 1D example with two classes, visualizing:
      - The prior P(y)
      - The class-conditional likelihoods P(x | y)
      - The resulting posterior P(y | x)

    This is the intuition behind why Bayes' theorem works: the posterior
    shifts away from the prior toward whichever class is more likely
    to generate the observed x.
    """
    np.random.seed(42)

    mu0, sigma0 = 1.0, 1.0
    mu1, sigma1 = 4.0, 1.2
    prior0, prior1 = 0.6, 0.4

    x = np.linspace(-2, 8, 500)
    p0 = norm.pdf(x, mu0, sigma0)
    p1 = norm.pdf(x, mu1, sigma1)

    evidence = p0 * prior0 + p1 * prior1
    post0 = p0 * prior0 / evidence
    post1 = p1 * prior1 / evidence

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        "Bayes' Theorem: how the prior and likelihood combine into the posterior",
        fontsize=12,
    )

    # --- Prior ---
    ax = axes[0]
    ax.bar([0, 1], [prior0, prior1], color=["steelblue", "coral"], alpha=0.8, width=0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["y = 0", "y = 1"])
    ax.set_title("Prior  P(y)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.0)
    for i, v in enumerate([prior0, prior1]):
        ax.text(i, v + 0.02, f"{v:.1f}", ha="center")

    # --- Likelihoods ---
    ax = axes[1]
    ax.plot(x, p0, color="steelblue", lw=2, label=r"$P(x \mid y=0)$")
    ax.plot(x, p1, color="coral", lw=2, label=r"$P(x \mid y=1)$")
    ax.axvline(3.0, color="gray", ls="--", lw=1)
    ax.text(3.05, 0.33, r"$x^* = 3$", fontsize=9)
    ax.set_title(r"Likelihood  $P(x \mid y)$")
    ax.set_xlabel("x")
    ax.legend(fontsize=9)

    # --- Posterior ---
    ax = axes[2]
    ax.plot(x, post0, color="steelblue", lw=2, label=r"$P(y=0 \mid x)$")
    ax.plot(x, post1, color="coral", lw=2, label=r"$P(y=1 \mid x)$")
    ax.axvline(
        x[np.argmax(np.abs(post0 - post1) < 0.01)],
        color="black",
        ls=":",
        lw=1.2,
        label="decision boundary",
    )
    ax.set_title(r"Posterior  $P(y \mid x)$")
    ax.set_xlabel("x")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(_IMG_DIR, "fig0_bayes_theorem.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved", path)


# ---------------------------------------------------------------------------
# fig1 — Class-conditional Gaussians overlaid on histograms
# ---------------------------------------------------------------------------


def fig1_class_conditionals():
    """
    Shows the learned per-feature Gaussians (the likelihood model) overlaid
    on histograms of the actual training data. Each subplot is one feature;
    each color is one class. This is what the model memorizes during fit().
    """
    np.random.seed(42)

    N = 300
    X0 = np.column_stack([np.random.normal(1.0, 1.0, N), np.random.normal(2.0, 0.8, N)])
    X1 = np.column_stack([np.random.normal(4.0, 1.2, N), np.random.normal(4.5, 0.6, N)])
    X = np.vstack([X0, X1])
    y = np.array([0] * N + [1] * N)

    means, variances, log_priors, classes = _fit_nb(X, y)

    feature_names = ["Feature 1", "Feature 2"]
    colors = ["steelblue", "coral"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        "Learned class-conditional Gaussians vs. training data histograms", fontsize=12
    )

    for j, ax in enumerate(axes):
        for k, cls in enumerate(classes):
            data = X[y == cls, j]
            ax.hist(
                data,
                bins=30,
                alpha=0.35,
                color=colors[k],
                density=True,
                label=f"class {cls} data",
            )
            xgrid = np.linspace(data.min() - 1, data.max() + 1, 300)
            ax.plot(
                xgrid,
                norm.pdf(xgrid, means[k, j], np.sqrt(variances[k, j])),
                color=colors[k],
                lw=2,
                label=f"class {cls} fit",
            )
        ax.set_title(feature_names[j])
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(_IMG_DIR, "fig1_class_conditionals.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved", path)


# ---------------------------------------------------------------------------
# fig2 — Decision boundary comparison: NB vs. logistic regression
# ---------------------------------------------------------------------------


def fig2_decision_boundary():
    """
    Compares NB's quadratic boundary (when class variances differ) with
    logistic regression's linear boundary on the same dataset.

    The left panel has equal class variances (NB → linear boundary).
    The right panel has unequal class variances (NB → curved boundary).
    """
    np.random.seed(42)

    N = 200

    def _make_data(sigma0, sigma1):
        X0 = np.random.multivariate_normal([0, 0], sigma0 * np.eye(2), N)
        X1 = np.random.multivariate_normal([4, 4], sigma1 * np.eye(2), N)
        X = np.vstack([X0, X1])
        y = np.array([0] * N + [1] * N)
        return X, y

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = [
        "Equal class variances\n(NB boundary ≈ linear)",
        "Unequal class variances\n(NB boundary is quadratic)",
    ]
    sigma_pairs = [(1.0, 1.0), (0.6, 2.0)]

    for ax, title, (s0, s1) in zip(axes, titles, sigma_pairs):
        X, y = _make_data(s0, s1)

        # Fit NB
        means, variances, log_priors, classes = _fit_nb(X, y)

        # Fit logistic regression
        w, b = _fit_logreg(X, y)

        # Decision grid
        x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300)
        x2 = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300)
        xx1, xx2 = np.meshgrid(x1, x2)
        Xgrid = np.column_stack([xx1.ravel(), xx2.ravel()])

        nb_log_post = _nb_log_posterior(Xgrid, means, variances, log_priors)
        nb_pred = (nb_log_post[:, 1] > nb_log_post[:, 0]).reshape(xx1.shape)

        lr_pred = (_sigmoid(Xgrid @ w + b) >= 0.5).reshape(xx1.shape)

        ax.contourf(xx1, xx2, nb_pred, alpha=0.15, cmap="RdBu")
        ax.contour(
            xx1, xx2, nb_pred, levels=[0.5], colors="navy", linewidths=2, linestyles="-"
        )
        ax.contour(
            xx1,
            xx2,
            lr_pred.astype(float),
            levels=[0.5],
            colors="darkred",
            linewidths=2,
            linestyles="--",
        )

        ax.scatter(
            X[y == 0, 0],
            X[y == 0, 1],
            s=12,
            alpha=0.5,
            color="steelblue",
            label="class 0",
        )
        ax.scatter(
            X[y == 1, 0], X[y == 1, 1], s=12, alpha=0.5, color="coral", label="class 1"
        )

        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="navy", lw=2, label="NB boundary"),
            Line2D([0], [0], color="darkred", lw=2, ls="--", label="Logistic boundary"),
        ]
        ax.legend(handles=legend_elements, fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    fig.suptitle("Naive Bayes vs. Logistic Regression Decision Boundaries", fontsize=12)
    plt.tight_layout()
    path = os.path.join(_IMG_DIR, "fig2_decision_boundary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved", path)


# ---------------------------------------------------------------------------
# fig3 — Accuracy vs. feature correlation (testing the naive assumption)
# ---------------------------------------------------------------------------


def fig3_naive_assumption():
    """
    Sweeps the correlation coefficient rho between two features and measures
    NB accuracy vs. logistic regression accuracy. As rho grows, the features
    become more linearly dependent, violating the naive assumption. NB degrades
    more than logistic regression.
    """
    np.random.seed(42)

    N = 400
    n_trials = 8
    rhos = np.linspace(0, 0.95, 15)
    nb_accs = np.zeros(len(rhos))
    lr_accs = np.zeros(len(rhos))

    for i, rho in enumerate(rhos):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        trial_nb = []
        trial_lr = []
        for _ in range(n_trials):
            X0 = np.random.multivariate_normal([0, 0], cov, N // 2)
            X1 = np.random.multivariate_normal([2, 2], cov, N // 2)
            X = np.vstack([X0, X1])
            y = np.array([0] * (N // 2) + [1] * (N // 2))

            means, variances, log_priors, classes = _fit_nb(X, y)
            nb_log_post = _nb_log_posterior(X, means, variances, log_priors)
            nb_pred = classes[np.argmax(nb_log_post, axis=1)]
            trial_nb.append(np.mean(nb_pred == y))

            w, b = _fit_logreg(X, y)
            lr_pred = (_sigmoid(X @ w + b) >= 0.5).astype(int)
            trial_lr.append(np.mean(lr_pred == y))

        nb_accs[i] = np.mean(trial_nb)
        lr_accs[i] = np.mean(trial_lr)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rhos, nb_accs, "o-", color="steelblue", lw=2, label="Gaussian Naive Bayes")
    ax.plot(rhos, lr_accs, "s--", color="coral", lw=2, label="Logistic Regression")
    ax.set_xlabel(r"Feature correlation $\rho$", fontsize=11)
    ax.set_ylabel("Training accuracy", fontsize=11)
    ax.set_title(
        "Accuracy vs. Feature Correlation\n(naive assumption is violated as ρ → 1)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(_IMG_DIR, "fig3_naive_assumption.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved", path)


# ---------------------------------------------------------------------------
# fig4 — Multi-class classification
# ---------------------------------------------------------------------------


def fig4_multiclass():
    """
    Three-class example showing NB's natural multi-class support.
    Overlays the Gaussian contours for each class (the likelihood model)
    on the decision regions.
    """
    np.random.seed(42)

    N = 150
    centers = [(-3, 0), (3, 0), (0, 4)]
    covs = [
        np.array([[1.2, 0.4], [0.4, 0.8]]),
        np.array([[0.8, -0.3], [-0.3, 1.5]]),
        np.array([[1.0, 0.0], [0.0, 0.6]]),
    ]
    colors = ["steelblue", "coral", "seagreen"]

    Xs, ys = [], []
    for k, (c, cov) in enumerate(zip(centers, covs)):
        Xi = np.random.multivariate_normal(c, cov, N)
        Xs.append(Xi)
        ys.append(np.full(N, k))
    X = np.vstack(Xs)
    y = np.concatenate(ys)

    means, variances, log_priors, classes = _fit_nb(X, y)

    x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 400)
    x2 = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 400)
    xx1, xx2 = np.meshgrid(x1, x2)
    Xgrid = np.column_stack([xx1.ravel(), xx2.ravel()])

    nb_log_post = _nb_log_posterior(Xgrid, means, variances, log_priors)
    nb_pred = np.argmax(nb_log_post, axis=1).reshape(xx1.shape)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Decision regions
    cmap = plt.cm.RdYlGn
    ax.contourf(
        xx1,
        xx2,
        nb_pred,
        levels=[-0.5, 0.5, 1.5, 2.5],
        alpha=0.15,
        colors=["steelblue", "coral", "seagreen"],
    )
    ax.contour(xx1, xx2, nb_pred, levels=[0.5, 1.5], colors="black", linewidths=1.2)

    # Per-class Gaussian contours (from learned diagonal covariance)
    for k in range(len(classes)):
        cov_k = np.diag(variances[k])
        rv = multivariate_normal(mean=means[k], cov=cov_k)
        z = rv.pdf(np.column_stack([xx1.ravel(), xx2.ravel()])).reshape(xx1.shape)
        levels = sorted(np.exp(-0.5 * np.array([1.0, 2.0]) ** 2) * z.max())
        ax.contour(
            xx1,
            xx2,
            z,
            levels=levels,
            colors=[colors[k]],
            linewidths=1.5,
            linestyles="--",
        )

    # Scatter
    for k, color in enumerate(colors):
        mask = y == k
        ax.scatter(
            X[mask, 0], X[mask, 1], s=14, alpha=0.6, color=color, label=f"class {k}"
        )

    ax.set_title(
        "Multi-class Gaussian Naive Bayes\n(dashed = 1σ / 2σ Gaussian contours per class)",
        fontsize=12,
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(_IMG_DIR, "fig4_multiclass.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fig0_bayes_theorem()
    fig1_class_conditionals()
    fig2_decision_boundary()
    fig3_naive_assumption()
    fig4_multiclass()
    print("All figures saved to", _IMG_DIR)
