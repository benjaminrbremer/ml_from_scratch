# Gaussian Naive Bayes

**Implementation:** [`naive_bayes.py`](./naive_bayes.py) · [`distributions.py`](./distributions.py)  
**See also:** [`logistic_regression.md`](../logistic_regression.md) · [`linear_regression.md`](../../regression/linear_regression.md)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Bayes' Theorem](#2-bayes-theorem)
3. [The Naive Assumption](#3-the-naive-assumption)
4. [Gaussian Naive Bayes](#4-gaussian-naive-bayes)
5. [Parameter Estimation (MLE)](#5-parameter-estimation-mle)
6. [Prediction in Log Space](#6-prediction-in-log-space)
7. [Multi-class Extension](#7-multi-class-extension)
8. [Variance Smoothing](#8-variance-smoothing)
9. [Worked Example](#9-worked-example)
10. [Comparison with Logistic Regression](#10-comparison-with-logistic-regression)
11. [Extensions](#11-extensions)
12. [Further Reading](#12-further-reading)

---

## 1. Introduction

Every classifier in this repo so far — linear and logistic regression — is a **discriminative** model. It directly learns $P(y \mid x)$: the probability of a label given the input. The model has no opinion about where inputs come from; it only learns the boundary between classes.

**Naive Bayes** flips the framing. It is a **generative** model: it asks "what distribution would *generate* the data for each class?" Once you know that, Bayes' theorem tells you which class most likely generated any new observation.

This distinction matters in practice:

| | Discriminative | Generative |
|---|---|---|
| **Learns** | $P(y \mid x)$ directly | $P(x \mid y)$ and $P(y)$ |
| **Uses** | Boundary between classes | A model of each class's data |
| **Training** | Iterative (gradient descent) | Closed-form (one pass) |
| **Examples** | Logistic regression, SVM | Naive Bayes, LDA, HMM |

Naive Bayes is notable for two properties that make it a great learning target:

- **No iteration.** Fitting is a single pass over the training data to compute means, variances, and class counts. There is no learning rate to tune and no convergence to wait for.
- **Surprisingly effective.** The "naive" independence assumption is almost always violated, yet Naive Bayes often matches or beats more sophisticated classifiers, especially with small training sets or high-dimensional features (like text).

---

## 2. Bayes' Theorem

The foundation is Bayes' theorem, derived from the definition of conditional probability:

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

Applied to classification — where $A = y$ (the class label) and $B = x$ (the observed features):

$$
\boxed{P(y \mid x) = \frac{P(x \mid y) \cdot P(y)}{P(x)}}
$$

Each term has a name:

| Term | Name | Meaning |
|---|---|---|
| $P(y \mid x)$ | **Posterior** | Probability of class $y$ *given* the input $x$. This is what we want. |
| $P(x \mid y)$ | **Likelihood** | Probability of seeing $x$ *if* the class were $y$. |
| $P(y)$ | **Prior** | Probability of class $y$ before seeing any data. |
| $P(x)$ | **Evidence** | Probability of seeing $x$ at all, regardless of class. |

### Why we can ignore the evidence

$P(x)$ does not depend on the class $y$. When we compare two classes:

$$
\frac{P(y = 0 \mid x)}{P(y = 1 \mid x)} = \frac{P(x \mid y=0) \cdot P(y=0)}{P(x \mid y=1) \cdot P(y=1)}
$$

The $P(x)$ in the denominator of each posterior cancels. For *prediction* (picking the class with the highest probability), we only need the numerator:

$$
\hat{y} = \arg\max_k \; P(x \mid y = k) \cdot P(y = k)
$$

This is the core prediction rule. The evidence only matters if you need *calibrated* probabilities (where the posteriors must sum to 1), which is why `predict_proba` includes a normalization step.

---

## 3. The Naive Assumption

The key computational challenge is modeling $P(x \mid y = k)$ for a feature vector $x \in \mathbb{R}^n$. The joint distribution over $n$ features is in general very complex: every feature could correlate with every other feature differently in each class.

The **naive assumption** sidesteps this by assuming the features are **conditionally independent** given the class:

$$
P(x \mid y = k) = \prod_{j=1}^{n} P(x_j \mid y = k)
$$

Instead of modeling one complex $n$-dimensional distribution per class, we model $n$ independent one-dimensional distributions — one per feature. The parameter count drops from $O(2^n)$ to $O(n)$.

### When is this assumption wrong?

Almost always. Consider classifying emails as spam or not:
- Words "free" and "money" are correlated: spam often contains both.
- Naive Bayes treats them as independent, so seeing both has double the effect of seeing either alone.

### Why does it still work?

Even when the probabilities are wrong, the *argmax* can still be correct. Naive Bayes tends to be overconfident (posterior probabilities push toward 0 and 1), but the class it picks can still be the right one. In the high-variance, small-data regime, the reduced parameter count (from the naive assumption) can actually *help* by avoiding overfitting.

---

## 4. Gaussian Naive Bayes

Once we've accepted the naive independence assumption, we need to choose a family of distributions for each $P(x_j \mid y = k)$. For continuous features, the natural choice is the Gaussian (normal) distribution:

$$
\boxed{x_j \mid y = k \;\sim\; \mathcal{N}(\mu_{kj},\; \sigma_{kj}^2)}
$$

The Gaussian PDF for one feature:

$$
P(x_j \mid y = k) = \frac{1}{\sqrt{2\pi\sigma_{kj}^2}} \exp\!\left(-\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right)
$$

Under the naive assumption, the joint likelihood for all features is the product:

$$
P(x \mid y = k) = \prod_{j=1}^{n} \frac{1}{\sqrt{2\pi\sigma_{kj}^2}} \exp\!\left(-\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right)
$$

There is one pair of parameters $(\mu_{kj}, \sigma_{kj}^2)$ per class $k$ and feature $j$. For $K$ classes and $n$ features, the model has $2Kn$ parameters total — all estimated directly from training data.

> **Why Gaussian?** Many real-valued features (heights, temperatures, pixel intensities) are approximately Gaussian within a class. Even when they're not, the Gaussian is often a reasonable proxy, especially in lower dimensions. See [Section 11](#11-extensions) for alternatives.

---

## 5. Parameter Estimation (MLE)

How should we set $\mu_{kj}$ and $\sigma_{kj}^2$? The standard answer is **maximum likelihood estimation (MLE)**: find the parameter values that make the observed training data most probable.

### Class prior

The prior $P(y = k)$ is simply the fraction of training samples with label $k$:

$$
\boxed{\hat{P}(y = k) = \frac{N_k}{N}}
$$

where $N_k$ is the number of class-$k$ samples and $N$ is the total. This is the MLE for a Bernoulli/categorical distribution.

### Gaussian parameters

For each class $k$ and feature $j$, we fit a Gaussian to the values of feature $j$ among class-$k$ samples:

$$
\boxed{\hat{\mu}_{kj} = \frac{1}{N_k} \sum_{i:\, y_i = k} x_{ij}}
$$

$$
\boxed{\hat{\sigma}_{kj}^2 = \frac{1}{N_k} \sum_{i:\, y_i = k} (x_{ij} - \hat{\mu}_{kj})^2}
$$

These are the sample mean and (biased) sample variance. They maximize the log-likelihood of a Gaussian for that feature within that class. The derivation:

**Log-likelihood of a Gaussian for one feature, one class:**

$$
\ell(\mu, \sigma^2) = \sum_{i:\, y_i = k} \log P(x_{ij} \mid \mu, \sigma^2)
= -\frac{N_k}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i:\, y_i=k}(x_{ij} - \mu)^2
$$

**Setting $\partial\ell/\partial\mu = 0$:**

$$
\frac{\partial\ell}{\partial\mu} = \frac{1}{\sigma^2}\sum_{i:\,y_i=k}(x_{ij} - \mu) = 0
\;\implies\; \mu = \frac{1}{N_k}\sum_{i:\,y_i=k} x_{ij}
$$

**Setting $\partial\ell/\partial(\sigma^2) = 0$:**

$$
\frac{\partial\ell}{\partial\sigma^2} = -\frac{N_k}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i:\,y_i=k}(x_{ij}-\mu)^2 = 0
\;\implies\; \sigma^2 = \frac{1}{N_k}\sum_{i:\,y_i=k}(x_{ij}-\mu)^2
$$

These are exactly the sample mean and sample variance — which should feel intuitive. The MLE for a Gaussian is just "compute the empirical moments."

---

## 6. Prediction in Log Space

At prediction time, we compute the log-posterior for each class:

$$
\log P(y = k \mid x) \propto \underbrace{\log P(y = k)}_{\text{log prior}} + \underbrace{\sum_{j=1}^{n} \log P(x_j \mid y = k)}_{\text{log likelihood}}
$$

Substituting the Gaussian log-PDF:

$$
\log P(x_j \mid y = k) = -\frac{1}{2}\left[\log(2\pi\sigma_{kj}^2) + \frac{(x_j - \mu_{kj})^2}{\sigma_{kj}^2}\right]
$$

So the full log-posterior is:

$$
\boxed{\log P(y = k \mid x) \propto \log\hat{P}(y=k) - \frac{1}{2}\sum_{j=1}^n \left[\log(2\pi\sigma_{kj}^2) + \frac{(x_j - \mu_{kj})^2}{\sigma_{kj}^2}\right]}
$$

### Why log space?

The likelihood is a product of $n$ Gaussian PDFs. For $n = 100$ features and a PDF value of $0.4$ per feature, the raw product is $0.4^{100} \approx 10^{-40}$ — below the range of `float64`, which bottoms out at roughly $10^{-308}$. Even for moderate $n$, the product underflows to zero before comparison is possible. Taking the log converts the product to a sum, keeping values in a numerically safe range.

### Log-sum-exp for normalized probabilities

When you need normalized posteriors $P(y = k \mid x)$ that sum to 1 (not just the argmax), you need to divide by the evidence $P(x) = \sum_k P(x \mid y=k) P(y=k)$. In log space, this is:

$$
\log P(x) = \log \sum_{k} \exp\!\left(\log P(x \mid y=k) + \log P(y=k)\right)
$$

The **log-sum-exp trick** prevents overflow/underflow by subtracting the maximum log-posterior before exponentiating:

$$
\log \sum_k e^{a_k} = c + \log \sum_k e^{a_k - c}, \quad c = \max_k a_k
$$

After subtracting $c$, the largest exponent is $e^0 = 1$, so all terms are in $[0, 1]$ and no overflow occurs.

---

## 7. Multi-class Extension

Naive Bayes extends to $K > 2$ classes without modification — it is inherently a multi-class model. The log-posterior formula is the same; we just compute it for each of the $K$ classes and take the argmax:

$$
\hat{y} = \arg\max_{k \in \{0, 1, \ldots, K-1\}} \left[\log\hat{P}(y=k) + \sum_{j=1}^{n} \log P(x_j \mid y=k)\right]
$$

Compare this to logistic regression, which is binary-native. Extending logistic regression to $K$ classes requires reformulation (softmax / multinomial logistic). Naive Bayes just adds more classes to the argmax.

---

## 8. Variance Smoothing

**The problem:** if a feature has the same value for all training samples within some class, the MLE variance is zero:

$$
\sigma_{kj}^2 = 0 \;\implies\; \log(2\pi\sigma_{kj}^2) = -\infty, \;\; \frac{(x_j - \mu_{kj})^2}{\sigma_{kj}^2} = \pm\infty
$$

This collapses the entire prediction for any test sample where $x_j \ne \mu_{kj}$.

**The fix:** add a small epsilon to every class variance before storing:

$$
\hat{\sigma}_{kj}^2 \;\leftarrow\; \hat{\sigma}_{kj}^2 + \varepsilon \cdot \max_{k', j'}\hat{\sigma}_{k'j'}^2
$$

Scaling $\varepsilon$ by the largest observed variance (rather than using a fixed constant) keeps the smoothing proportional to the data's natural scale. For $\varepsilon = 10^{-9}$ and a largest variance of $1.0$, the added term is $10^{-9}$ — essentially invisible compared to any real variance, but enough to prevent division by zero.

This is the same strategy as scikit-learn's `GaussianNB(var_smoothing=1e-9)`.

---

## 9. Worked Example

### Setup

Suppose we have 6 training samples, 2 features, 2 classes:

| Sample | $x_1$ | $x_2$ | $y$ |
|---|---|---|---|
| 1 | 1.0 | 2.0 | 0 |
| 2 | 1.5 | 1.8 | 0 |
| 3 | 0.8 | 2.3 | 0 |
| 4 | 4.0 | 5.5 | 1 |
| 5 | 3.5 | 5.2 | 1 |
| 6 | 4.2 | 4.9 | 1 |

### Step 1: Class priors

$N_0 = 3, \; N_1 = 3, \; N = 6$

$$
\hat{P}(y = 0) = 3/6 = 0.5, \quad \hat{P}(y = 1) = 3/6 = 0.5
$$
$$
\log\hat{P}(y = 0) = \log 0.5 \approx -0.693, \quad \log\hat{P}(y = 1) \approx -0.693
$$

Equal priors here — the priors cancel in the argmax, so the decision reduces to likelihood alone.

### Step 2: Class means

$$
\hat{\mu}_{0,1} = (1.0 + 1.5 + 0.8)/3 = 1.1, \quad \hat{\mu}_{0,2} = (2.0 + 1.8 + 2.3)/3 = 2.033
$$
$$
\hat{\mu}_{1,1} = (4.0 + 3.5 + 4.2)/3 = 3.9, \quad \hat{\mu}_{1,2} = (5.5 + 5.2 + 4.9)/3 = 5.2
$$

### Step 3: Class variances

$$
\hat{\sigma}_{0,1}^2 = \frac{(1.0-1.1)^2 + (1.5-1.1)^2 + (0.8-1.1)^2}{3} = \frac{0.01 + 0.16 + 0.09}{3} \approx 0.087
$$
$$
\hat{\sigma}_{0,2}^2 = \frac{(2.0-2.033)^2 + (1.8-2.033)^2 + (2.3-2.033)^2}{3} \approx 0.036
$$
$$
\hat{\sigma}_{1,1}^2 \approx 0.087, \quad \hat{\sigma}_{1,2}^2 \approx 0.060
$$

### Step 4: Predict a new point $x^* = [2.0, 3.5]$

We compute $\log P(x^* \mid y = k) + \log P(y = k)$ for $k \in \{0, 1\}$.

**For class 0:**
$$
\log P(x_1^* \mid y=0) = -\frac{1}{2}\left[\log(2\pi \cdot 0.087) + \frac{(2.0 - 1.1)^2}{0.087}\right] \approx -\frac{1}{2}[{-0.58 + 9.31}] \approx -4.36
$$
$$
\log P(x_2^* \mid y=0) = -\frac{1}{2}\left[\log(2\pi \cdot 0.036) + \frac{(3.5 - 2.033)^2}{0.036}\right] \approx -\frac{1}{2}[{-1.62 + 59.9}] \approx -29.1
$$
$$
\log P(x^* \mid y=0) + \log P(y=0) \approx -4.36 + (-29.1) + (-0.693) \approx -34.2
$$

**For class 1:**
$$
\log P(x_1^* \mid y=1) = -\frac{1}{2}\left[\log(2\pi \cdot 0.087) + \frac{(2.0 - 3.9)^2}{0.087}\right] \approx -\frac{1}{2}[{-0.58 + 41.5}] \approx -20.5
$$
$$
\log P(x_2^* \mid y=1) = -\frac{1}{2}\left[\log(2\pi \cdot 0.060) + \frac{(3.5 - 5.2)^2}{0.060}\right] \approx -\frac{1}{2}[{-1.06 + 48.2}] \approx -23.6
$$
$$
\log P(x^* \mid y=1) + \log P(y=1) \approx -20.5 + (-23.6) + (-0.693) \approx -44.8
$$

**Decision:** $-34.2 > -44.8$, so $\hat{y} = 0$. The point $[2.0, 3.5]$ is closer (in the Gaussian sense) to the class-0 cluster.

### Verify with the implementation

```python
import numpy as np
from ml_from_scratch.classification.naive_bayes.naive_bayes import gaussian_naive_bayes

X = np.array([[1.0, 2.0], [1.5, 1.8], [0.8, 2.3],
              [4.0, 5.5], [3.5, 5.2], [4.2, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1])

nb = gaussian_naive_bayes(n_features=2)
nb.fit(X, y)

print("Means:\n", nb.means_)
# [[1.1   2.033]
#  [3.9   5.2  ]]

print("Predicted class for [2.0, 3.5]:", nb.predict(np.array([2.0, 3.5])))
# [0]
```

---

## 10. Comparison with Logistic Regression

Both models perform binary (and multi-class) classification, but their assumptions and training procedures differ fundamentally.

| | Gaussian Naive Bayes | Logistic Regression |
|---|---|---|
| **Type** | Generative | Discriminative |
| **Learns** | $P(x \mid y)$ and $P(y)$ | $P(y \mid x)$ directly |
| **Training** | Closed-form, one pass | Gradient descent, iterative |
| **Parameters** | $2Kn$ (means + variances) | $n + 1$ (weights + bias) |
| **Assumption** | Features independent given class; Gaussian within class | Linear decision boundary in feature space |
| **Decision boundary** | Quadratic (elliptical), or linear if class variances equal | Linear hyperplane |
| **Small data** | Works well (few parameters relative to expressive power) | May overfit without regularization |
| **Feature correlation** | Degrades (naive assumption violated) | Handles naturally |
| **Calibration** | Overconfident posteriors | Well-calibrated with sufficient data |

### Decision boundary shape

Logistic regression's boundary is always the hyperplane $x \cdot w + b = 0$ — flat by construction. Gaussian NB's boundary is implicitly defined by the set of $x$ where the two log-posteriors are equal. When the two classes have **different** covariance matrices, this locus is a *quadratic* surface (an ellipse in 2D). When they have **equal** covariance, it degenerates to a linear boundary identical in form to logistic regression's.

### When to prefer Naive Bayes

- Very small training sets — fewer parameters to estimate
- Very high-dimensional inputs (e.g. text) — $O(n)$ parameters instead of $O(n^2)$
- Online / streaming learning — statistics can be updated incrementally
- Need a fast baseline before training a heavier model
- Features are approximately independent (sensor readings, some medical tests)

### When to prefer Logistic Regression

- Features are correlated — NB's independence assumption will hurt
- You need well-calibrated probabilities
- The true boundary is known to be approximately linear
- More training data is available — logistic regression's stronger assumptions become worthwhile

---

## 11. Extensions

### Bernoulli Naive Bayes

For **binary features** $x_j \in \{0, 1\}$ (e.g. "does word $j$ appear in this document?"), model each $P(x_j \mid y = k)$ as Bernoulli with parameter $p_{kj} = P(x_j = 1 \mid y = k)$:

$$
P(x_j \mid y = k) = p_{kj}^{x_j} (1 - p_{kj})^{1 - x_j}
$$

The MLE estimate is simply the fraction of class-$k$ samples where feature $j$ is 1: $\hat{p}_{kj} = N_{kj} / N_k$.

**Laplace smoothing** (add-one smoothing) prevents zero counts: replace $N_{kj} / N_k$ with $(N_{kj} + 1) / (N_k + 2)$. This is the discrete equivalent of variance smoothing.

### Multinomial Naive Bayes

For **count features** $x_j \in \{0, 1, 2, \ldots\}$ — most famously, word counts in bag-of-words text classification — model $P(x \mid y = k)$ as multinomial:

$$
P(x \mid y = k) \propto \prod_{j=1}^n \theta_{kj}^{x_j}
$$

where $\theta_{kj} = P(\text{word } j \mid \text{class } k)$ is estimated as the fraction of words in class-$k$ documents that are word $j$. With Laplace smoothing:

$$
\hat{\theta}_{kj} = \frac{N_{kj} + 1}{\sum_{j'} (N_{kj'} + 1)}
$$

Multinomial NB is among the strongest text classifiers despite its simplicity, and it is still widely used in spam filtering.

### Categorical Naive Bayes

For features that take one of $M$ discrete values (not necessarily $\{0, 1\}$), model each $P(x_j \mid y = k)$ as categorical. This is the most general discrete case.

### Relaxing the naive assumption: Bayesian networks

If you know which features are correlated, you can replace the naive (fully independent) assumption with a Bayesian network that explicitly models the conditional dependencies. This is exponentially more expensive in general but tractable when the dependency structure is sparse.

---

## 12. Further Reading

- **Bishop, *Pattern Recognition and Machine Learning*, Ch. 4** — Generative models for classification, discriminant functions, and the connection to Gaussian mixture models.
- **Mitchell, *Machine Learning*, Ch. 6** — The original textbook treatment of Naive Bayes for text classification.
- **Rennie et al., "Tackling the Poor Assumptions of Naive Bayes Text Classifiers"** (ICML 2003) — Why NB works so well for text despite violated assumptions.
- **Ng and Jordan, "On Discriminative vs. Generative Classifiers"** (NeurIPS 2001) — Formal analysis of when Naive Bayes (generative) beats logistic regression (discriminative) and vice versa.
- **scikit-learn docs: `GaussianNB`** — Reference implementation with variance smoothing and incremental fit.
