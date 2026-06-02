# K-Means Clustering — Math, Code, and Practice

A companion to [`k_means.py`](./k_means.py). The goal is to walk through
*why* the algorithm works, *how* the math translates into vectorized NumPy,
and *when* k-means is (and isn't) the right tool.

Math is rendered with LaTeX (`$...$`). GitHub, VS Code, and most modern
markdown viewers render this inline.

## Table of contents

1. [Supervised vs. unsupervised learning](#1-supervised-vs-unsupervised-learning)
2. [The model: centroids and Voronoi regions](#2-the-model-centroids-and-voronoi-regions)
3. [Objective function: WCSS](#3-objective-function-wcss)
4. [Why the mean minimizes WCSS](#4-why-the-mean-minimizes-wcss)
5. [The algorithm](#5-the-algorithm)
6. [Convergence guarantee](#6-convergence-guarantee)
7. [Initialization and K-means++](#7-initialization-and-k-means)
8. [Vectorized NumPy implementation](#8-vectorized-numpy-implementation)
9. [Worked example](#9-worked-example)
10. [Choosing K](#10-choosing-k)
11. [Limitations](#11-limitations)
12. [Extensions](#12-extensions)
13. [Further reading](#13-further-reading)

---

## 1. Supervised vs. unsupervised learning

Every model so far — linear regression and logistic regression — is
**supervised**: each training sample comes with a label $y$ telling the model
what the right answer is. The model's parameters are tuned to minimize the
gap between its predictions and those labels.

**Unsupervised learning** removes the labels entirely. The input is just a
set of points $X = \{x_1, \ldots, x_N\}$, and the goal is to discover
*structure* in the data itself rather than to predict an externally-supplied
target.

K-means is the simplest and most widely-used unsupervised algorithm.
Its task is **clustering**: partition the $N$ points into $K$ groups
(clusters) such that points in the same group are similar to each other
and dissimilar from points in other groups. "Similar" is measured by
Euclidean distance.

---

## 2. The model: centroids and Voronoi regions

K-means represents each cluster $k$ by a single point $\mu_k \in \mathbb{R}^n$
called the **centroid** (or prototype). Given a fixed set of centroids
$\{\mu_1, \ldots, \mu_K\}$, every data point is assigned to whichever
centroid is closest:

$$\text{label}_i = \arg\min_{k \in \{1,\ldots,K\}} \| x_i - \mu_k \|^2$$

The regions of space that map to each centroid form a **Voronoi diagram**:
the boundary between cluster $j$ and cluster $k$ is the perpendicular
bisector of the segment $\overline{\mu_j \mu_k}$, which is the set of
points equidistant from both. The full partition is:

$$C_k = \{ x_i : \| x_i - \mu_k \|^2 \leq \| x_i - \mu_j \|^2 \text{ for all } j \}$$

Each cluster $C_k$ is a convex polytope. Together the $K$ polytopes tile
$\mathbb{R}^n$ without overlap. This means k-means always produces
**convex, roughly spherical** decision regions — a fact we'll revisit in
the limitations section.

---

## 3. Objective function: WCSS

The quantity k-means minimizes is the **within-cluster sum of squares**
(WCSS), also called the **inertia**:

$$J(\mu_1, \ldots, \mu_K) = \sum_{k=1}^{K} \sum_{i \in C_k} \| x_i - \mu_k \|^2$$

This is a sum over all $N$ points of the squared distance to the nearest
centroid. It measures how "tight" or "compact" the clusters are. Perfect
clustering (every point exactly at its centroid) gives $J = 0$.

Notice that $J$ depends on both the centroid positions *and* the cluster
assignments, and the two are coupled: the assignments depend on the
centroids, and the optimal centroid for a fixed assignment is the mean
of the assigned points (proved in the next section). This coupling is
what makes exact minimization of $J$ NP-hard in general
(the problem is equivalent to an integer program with exponentially many
possible assignments). K-means solves it approximately via alternating
optimization.

---

## 4. Why the mean minimizes WCSS

Given a fixed assignment of points to clusters, what centroid $\mu_k$
minimizes the within-cluster sum of squares for cluster $k$?

$$\text{minimize}_{\mu_k} \; f(\mu_k) = \sum_{i \in C_k} \| x_i - \mu_k \|^2$$

Expanding the squared norm component-wise for feature $j$:

$$f(\mu_k) = \sum_{i \in C_k} \sum_{j=1}^{n} (x_{ij} - \mu_{kj})^2$$

Since the sum over $j$ separates across coordinates, we can minimize
each feature independently. For a single feature $j$, set the derivative
to zero:

$$\frac{\partial f}{\partial \mu_{kj}} = \sum_{i \in C_k} 2(x_{ij} - \mu_{kj}) \cdot (-1) = 0$$

$$\Rightarrow \sum_{i \in C_k} (x_{ij} - \mu_{kj}) = 0$$

$$\Rightarrow \mu_{kj} = \frac{1}{|C_k|} \sum_{i \in C_k} x_{ij}$$

The minimizer is the **sample mean** of the cluster. In vector form:

$$\mu_k^* = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$$

The second derivative is $2|C_k| > 0$, confirming this is a minimum.
This is why the update step of k-means simply takes the mean — it's not
a heuristic, it's the exact closed-form solution to the optimization
problem "given these assignments, what centroids minimize WCSS?"

---

## 5. The algorithm

The k-means algorithm alternates between two steps:

```
Initialize: choose K starting centroids mu_1, ..., mu_K (see Section 7)

Repeat until convergence:
    Assignment step (E-step):
        For each i = 1..N:
            label_i = argmin_k || x_i - mu_k ||^2

    Update step (M-step):
        For each k = 1..K:
            mu_k = mean of { x_i : label_i = k }
```

The labels "E-step" and "M-step" come from the **Expectation-Maximization
(EM) algorithm**, which k-means is a hard-assignment special case of
(see [Extensions](#12-extensions)). In the E-step we fix the parameters
and optimize the assignments; in the M-step we fix the assignments and
optimize the parameters (centroids).

**In code** (from [`k_means.py`](./k_means.py)):

```python
# Assignment step — vectorized via broadcasting
def predict(self, X):
    sq_dists = self._sq_distances(X, self.centroids)  # (N, K)
    return np.argmin(sq_dists, axis=1)                 # (N,)

# Update step — loop over K clusters (not N points)
for j in range(self.k):
    assigned = X[labels == j]
    new_centroids[j] = assigned.mean(axis=0) if len(assigned) > 0 else self.centroids[j]
```

The loop is over $K$ (number of clusters) rather than $N$ (number of
samples), so the Python-level overhead is $O(K)$, not $O(N)$. The
expensive inner work — the mean — uses NumPy and runs at C speed.

---

## 6. Convergence guarantee

**Claim:** k-means always terminates in a finite number of steps.

**Proof sketch:**

1. **Monotone decrease:** The assignment step minimizes $J$ over all
   possible assignments (it assigns each point to its *nearest* centroid).
   The update step minimizes $J$ over all possible centroid positions
   (each centroid is set to the mean of its cluster, which Section 4
   shows is the minimizer). So each step either decreases $J$ or leaves
   it unchanged.

2. **Bounded below:** $J \geq 0$ always.

3. **Finite state space:** There are only $K^N$ possible assignments of
   $N$ points to $K$ clusters. Monotone decrease means no assignment is
   ever revisited. Therefore the algorithm must terminate.

**Caveat:** The guarantee is that the algorithm terminates, *not* that it
finds the global minimum. K-means is not guaranteed to find the global
minimum of $J$; it converges to a local minimum whose quality depends
on the initialization. This is the main motivation for K-means++
(Section 7) and for running the algorithm multiple times with different
random seeds.

---

## 7. Initialization and K-means++

The choice of starting centroids has a large effect on the final WCSS.
Two bad initializations that illustrate the problem:

- **Cluster collapse:** multiple initial centroids land in the same
  dense region, so nearby clusters "steal" each other's points and one
  true cluster gets missed entirely.
- **Outlier seeding:** an initial centroid lands on an outlier far from
  any cluster, so one centroid attracts almost no points and is wasted.

### Random initialization

The simplest strategy: choose $K$ distinct data points uniformly at
random as the starting centroids. Fast and simple, but offers no
protection against the failure modes above.

### K-means++ (Arthur & Vassilvitskii, 2007)

**Idea:** spread the initial centroids far apart by sampling the next
centroid proportionally to the squared distance to the nearest
already-chosen centroid. Points that are far from all existing centroids
get high probability; points already near a centroid get low probability.

**Algorithm:**

```
1. Choose c_1 uniformly at random from X.
2. For m = 2, ..., K:
   a. For each point x_i, compute:
          D(x_i) = min_{j < m} || x_i - c_j ||  (distance to nearest chosen centroid)
   b. Sample c_m from X with probability proportional to D(x_i)^2.
3. Run standard k-means from these K starting centroids.
```

The $D^2$ weighting (squaring the distance) makes the sampling more
aggressive: a point twice as far from the nearest centroid is four times
as likely to be chosen as the next centroid, pushing new centroids into
underrepresented regions.

**Theoretical guarantee:** K-means++ with standard k-means refinement
achieves an expected WCSS within $O(\log K)$ of optimal:

$$\mathbb{E}[J_{\text{kmeans++}}] \leq 8(\ln K + 2) \cdot J_{\text{OPT}}$$

This is a worst-case bound; in practice the improvement is often much
larger, especially on datasets where clusters are well-separated.

**Implementation note** (from `_init_centroids`):

```python
# After choosing m centroids, compute D^2 for the (m+1)-th centroid:
sq_dists = self._sq_distances(X, centers_so_far)  # (N, m)
d2 = sq_dists.min(axis=1)                          # (N,) -- D(x)^2

probs = d2 / d2.sum()
next_idx = rng.choice(n, p=probs)
```

The call to `_sq_distances` reuses the vectorized broadcasting function
from the assignment step — no special-case code needed.

---

## 8. Vectorized NumPy implementation

### The distance matrix trick

The most performance-critical operation is computing squared distances
from every data point to every centroid. A naive Python loop would be:

```python
# SLOW — O(N*K) Python iterations
for i in range(N):
    for k in range(K):
        D[i, k] = np.sum((X[i] - centroids[k])**2)
```

The vectorized version uses NumPy **broadcasting** to compute all $N \times K$
distances in a single expression:

```python
# FAST — one C-level call
diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # shape (N, K, n)
D    = np.sum(diff**2, axis=2)                             # shape (N, K)
```

Here's what the index magic does:

| Expression | Shape | Meaning |
| :--------- | :---- | :------ |
| `X[:, np.newaxis, :]` | $(N, 1, n)$ | one data point per row, broadcast over K |
| `centroids[np.newaxis, :, :]` | $(1, K, n)$ | one centroid per "column", broadcast over N |
| `diff` | $(N, K, n)$ | `diff[i, k, j] = X[i, j] - centroids[k, j]` |
| `np.sum(diff**2, axis=2)` | $(N, K)$ | $D[i, k] = \|x_i - \mu_k\|^2$ |

NumPy's broadcast rules: when shapes don't match, dimensions of size 1
are stretched to match the other operand. So `(N, 1, n)` and `(1, K, n)`
broadcast to `(N, K, n)` without allocating the repeated copies.

### Centroid update

The update step loops over $K$ clusters and calls `.mean()` on each
subset. Because $K \ll N$ in typical use, the Python loop overhead is
negligible; the expensive work (summing and dividing up to $N/K$
values) happens inside NumPy.

```python
for j in range(self.k):
    assigned = X[labels == j]          # Boolean index: O(N) but in C
    new_centroids[j] = assigned.mean(axis=0)
```

Boolean indexing (`X[labels == j]`) is a vectorized operation that
creates a view (or copy) without a Python loop over $N$.

---

## 9. Worked example

Three well-separated clusters in 2D with $N = 9$ points and $K = 3$.

**Data:**

| Point | $x_1$ | $x_2$ | True cluster |
| :---- | ----: | ----: | :----------- |
| $a$   | 1     | 1     | Red          |
| $b$   | 1.5   | 2     | Red          |
| $c$   | 3     | 4     | Red          |
| $d$   | 5     | 7     | Blue         |
| $e$   | 3.5   | 5     | Blue         |
| $f$   | 4.5   | 5     | Blue         |
| $g$   | 3.5   | 3     | Green        |
| $h$   | 4     | 3     | Green        |
| $i$   | 4.5   | 2.5   | Green        |

**Initialization (K-means++):**

Suppose the seeding picks $\mu_1 = a$, $\mu_2 = d$, $\mu_3 = i$
(these are far apart — a good start for K-means++).

**Iteration 1 — Assignment:**

Compute squared distances from each point to each centroid.
For point $c = (3, 4)$:

$$\|c - \mu_1\|^2 = (3-1)^2 + (4-1)^2 = 4 + 9 = 13$$
$$\|c - \mu_2\|^2 = (3-5)^2 + (4-7)^2 = 4 + 9 = 13$$
$$\|c - \mu_3\|^2 = (3-4.5)^2 + (4-2.5)^2 = 2.25 + 2.25 = 4.5$$

Point $c$ goes to cluster 3 (nearest to $\mu_3 = i$).

After assigning all points: $C_1 = \{a, b\}$, $C_2 = \{d, e, f\}$, $C_3 = \{c, g, h, i\}$.

**Iteration 1 — Update:**

$$\mu_1 \leftarrow \text{mean}(\{(1,1),(1.5,2)\}) = (1.25, 1.5)$$
$$\mu_2 \leftarrow \text{mean}(\{(5,7),(3.5,5),(4.5,5)\}) = (4.33, 5.67)$$
$$\mu_3 \leftarrow \text{mean}(\{(3,4),(3.5,3),(4,3),(4.5,2.5)\}) = (3.75, 3.125)$$

**Iteration 2 — Assignment:**

Recompute with updated centroids. For point $c = (3, 4)$:

$$\|c - \mu_1\|^2 = (3-1.25)^2 + (4-1.5)^2 = 3.0625 + 6.25 = 9.31$$
$$\|c - \mu_2\|^2 = (3-4.33)^2 + (4-5.67)^2 = 1.77 + 2.79 = 4.56$$
$$\|c - \mu_3\|^2 = (3-3.75)^2 + (4-3.125)^2 = 0.5625 + 0.766 = 1.33$$

Point $c$ stays in cluster 3.

In this example the assignments don't change from iteration 1 to 2,
so the algorithm has converged after a single iteration. The centroids
shift from iteration 1 to 2 (the update moves them closer to the true
cluster centers), but since labels stabilize, we stop.

**Final inertia:**

$$J = \sum_{i \in C_1} \|x_i - \mu_1\|^2 + \sum_{i \in C_2} \|x_i - \mu_2\|^2 + \sum_{i \in C_3} \|x_i - \mu_3\|^2$$

Each term is the sum of squared distances within the cluster — a concrete
measure of how tight the clusters are.

---

## 10. Choosing K

K-means requires you to specify $K$ upfront, but often you don't know
the "true" number of clusters in advance. Two widely-used heuristics:

### The elbow method

Run k-means for $K = 1, 2, \ldots, K_{\max}$ and plot inertia vs. $K$.

As $K$ increases, inertia always decreases (more clusters can fit the
data more tightly). At $K = N$ each point is its own cluster and $J = 0$.
The "elbow" — where the rate of decrease sharply slows — is a natural
candidate for $K$.

The elbow is often ambiguous in practice (the curve can be smooth with
no clear kink), but it's a good starting point.

### Silhouette score

For each point $x_i$, define:

- $a_i$: mean distance to all other points in the same cluster
  (measures how well $x_i$ fits its own cluster).
- $b_i$: mean distance to all points in the nearest *other* cluster
  (measures how well-separated $x_i$ is from other clusters).

The **silhouette coefficient** for point $i$:

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \in [-1, 1]$$

A value near $+1$ means $x_i$ is well-matched to its cluster and
poorly-matched to neighboring clusters (good). Near $0$ means $x_i$
sits on the boundary between two clusters. Near $-1$ means $x_i$ might
belong to the wrong cluster.

The **mean silhouette** over all points gives a single quality score per
$K$. Unlike inertia, a higher silhouette score is better, and you can
often find a genuine maximum — making it a more principled criterion
than the elbow.

---

## 11. Limitations

### Local optima

K-means is not convex. The WCSS objective has many local minima,
and gradient descent — or alternating optimization as used here — is only
guaranteed to find a local minimum. Running k-means multiple times with
different random seeds and keeping the best result is standard practice.

### Fixed, known K

The algorithm requires $K$ as input. If you don't know $K$, you have to
try several values and use a secondary criterion (elbow, silhouette,
BIC/AIC for a GMM, domain knowledge) to choose.

### Assumes isotropic, similarly-sized clusters

The Euclidean distance metric and the centroid representation implicitly
assume that clusters are roughly **spherical** (equal variance in every
direction) and of **similar size**. K-means will struggle with:

- **Elongated clusters** (ellipse-shaped) — the centroid may attract
  points from the wrong end of a neighboring ellipse. Gaussian mixture
  models (GMMs) can model arbitrary covariance shapes.
- **Very different cluster sizes** — a small dense cluster near a large
  sparse cluster may be swallowed by the large one.
- **Non-convex clusters** (e.g., two concentric rings) — because Voronoi
  regions are always convex, k-means cannot capture non-convex shapes.
  Kernel k-means or spectral clustering handle these cases.

### Sensitivity to scale

K-means uses raw Euclidean distance. If features have very different
scales (e.g., age in years vs. income in dollars), high-variance features
will dominate the distance and effectively ignore the other features.
**Always standardize features** (subtract mean, divide by standard
deviation) before running k-means.

---

## 12. Extensions

### Soft k-means / Gaussian mixture models (GMMs)

K-means makes *hard* assignments: each point belongs to exactly one
cluster. A natural soft version assigns each point a **probability** of
belonging to each cluster. When the clusters are modeled as Gaussians,
this is a **Gaussian mixture model** trained by the EM algorithm:

- **E-step:** compute posterior probabilities
  $r_{ik} = P(\text{cluster } k \mid x_i)$ (responsibilities).
- **M-step:** update the mean, covariance, and mixing weight of each
  Gaussian using the responsibilities as soft weights.

GMMs strictly generalize k-means: k-means is the limit of a GMM where
all covariances are constrained to $\sigma^2 I$ and $\sigma^2 \to 0$.

### Mini-batch k-means

On very large datasets, the full E-step (assigning all $N$ points) and
M-step (averaging all assigned points) become expensive. Mini-batch
k-means draws a random subset of points each iteration, performs the
E and M steps on the mini-batch, and accumulates a running mean for
each centroid. It converges to a slightly worse solution than full
k-means but is dramatically faster on large data.

### Kernel k-means

Replace the Euclidean distance with $\|k(x_i) - k(x_j)\|^2$ where $k$
is a nonlinear feature map (or equivalently, define the algorithm in
terms of a kernel function $K(x, x') = \langle k(x), k(x') \rangle$).
This allows k-means to find non-convex clusters by implicitly working
in a high-dimensional space. Spectral clustering is closely related —
it applies k-means to the eigenvectors of the data's graph Laplacian.

### K-medoids (PAM)

Instead of using the mean as the centroid, k-medoids constrains each
centroid to be an actual data point (a "medoid"). This makes the
algorithm robust to outliers (the mean can be dragged far from the
cluster by a single outlier; the medoid cannot). The trade-off is
a more expensive update step: finding the medoid requires computing
all pairwise distances within the cluster.

---

## 13. Further reading

- **Arthur, D. & Vassilvitskii, S. (2007).** "k-means++: The advantages of
  careful seeding." *SODA 2007.* The original K-means++ paper with the
  $O(\log K)$ approximation guarantee.
- **Bishop, C. (2006).** *Pattern Recognition and Machine Learning.*
  Chapter 9 covers the EM algorithm and its connection to k-means and GMMs
  in depth.
- **Murphy, K. (2012).** *Machine Learning: A Probabilistic Perspective.*
  Chapters 11 (mixture models) and 25 (clustering) give a broad view of
  the unsupervised landscape.
- **Sculley, D. (2010).** "Web-scale k-means clustering." *WWW 2010.*
  The mini-batch k-means algorithm and its convergence properties.
