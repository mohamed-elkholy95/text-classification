# 🧠 Core Concepts in Text Classification

This document provides educational deep-dives into the fundamental concepts
used throughout the text classification pipeline. Each section explains
*what* the concept is, *why* it works, and gives a concrete example.

---

## TF-IDF (Term Frequency – Inverse Document Frequency)

**What:** TF-IDF converts text documents into numerical vectors. For each word
in a document, it computes two things: *term frequency* (how often the word
appears in this document) and *inverse document frequency* (how rare the word
is across all documents). The product TF × IDF gives a single score per word.

**Why it works:** Words like "the" and "is" appear in almost every document,
so their IDF is near zero — they're effectively ignored. Words like "amazing"
or "terrible" appear in only a few documents about specific sentiments, so
their IDF is high. TF-IDF automatically downweights common noise and
amplifies discriminative signal, making it a remarkably effective baseline
feature representation.

**Example:** In a dataset of 1000 reviews, "product" appears in 900 reviews
(IDF ≈ 0.1), while "defective" appears in only 10 reviews (IDF ≈ 4.6).
A review mentioning "defective" three times gets a TF-IDF score of ~13.8
for that term — far more informative than "product" scoring ~0.3.

---

## Naive Bayes

**What:** Naive Bayes applies Bayes' theorem to compute P(class | text) by
estimating P(word | class) from the training data. It multiplies the
per-word probabilities together and picks the class with the highest
posterior. The "naive" part is the assumption that words are conditionally
independent given the class.

**Why it works:** Despite the independence assumption being blatantly false
(words like "not" and "good" are clearly dependent), Naive Bayes works
surprisingly well for text. This is because classification only requires
that the *argmax* of P(class | text) is correct — the individual probability
estimates can be wrong as long as the ranking is right. With high-dimensional
sparse text data, the estimation errors tend to cancel out.

**Example:** P(spam | "free", "money", "now") ∝ P(spam) × P("free"|spam) ×
P("money"|spam) × P("now"|spam). If those words appear frequently in spam
emails but rarely in ham, the product will be much higher for the spam class.

---

## SVM (Support Vector Machine)

**What:** A linear SVM finds the hyperplane (decision boundary) that
maximizes the *margin* — the distance between the hyperplane and the
nearest data points from each class (called support vectors). Data points
on the wrong side of the margin incur a penalty proportional to their
distance, controlled by the regularization parameter C.

**Why it works:** Maximizing the margin gives the classifier the largest
possible "buffer zone" for generalization. SVMs handle high-dimensional
sparse data (like TF-IDF) extremely well because the margin depends only
on the support vectors, not on the entire dataset. This makes SVMs robust
to noise and efficient even when the number of features exceeds the number
of samples.

**Kernel trick:** For non-linearly separable data, the kernel trick
implicitly maps features into a higher-dimensional space where a linear
boundary exists. Common kernels: RBF (radial basis function), polynomial.
For text with TF-IDF, linear kernels are typically best because the data
is already high-dimensional.

---

## Confidence Calibration

**What:** A classifier's raw output scores (logits, decision functions)
are not necessarily well-calibrated probabilities. A model outputting 0.8
confidence might only be correct 60% of the time. Calibration transforms
these scores so that the predicted probability matches the empirical
accuracy.

**Why it matters:** Calibrated probabilities are essential when you need
to make threshold-dependent decisions (e.g., "only act if confidence > 90%").
They also enable meaningful comparison of confidence across different models
and datasets, and are critical for risk-sensitive applications like medical
diagnosis or fraud detection.

**Methods:**
- **Platt scaling:** Fits a logistic sigmoid to map raw scores to
  probabilities. Fast, works well with small calibration sets, but can
  be too rigid for complex score distributions.
- **Isotonic regression:** Fits a non-parametric monotonically increasing
  step function. More flexible than Platt scaling but requires more
  calibration data to avoid overfitting.

---

## Cross-Validation

**What:** Cross-validation splits the dataset into k folds (typically 5 or
10), trains on k-1 folds, evaluates on the held-out fold, and repeats for
all k possible splits. The final metric is the average across all folds.

**Stratification:** Stratified k-fold ensures that each fold has the same
class distribution as the full dataset. This is critical for imbalanced
datasets where random splits might put all minority-class samples in one fold.

**Why it works:** A single train/test split gives one estimate of
generalization performance that depends heavily on *which* samples end up
in the test set. Cross-validation gives k estimates and averages them,
dramatically reducing variance. It also lets you use all data for both
training and evaluation — no sample is "wasted" as a permanent test holdout.

**When to use:** Always use CV for model selection (choosing hyperparameters,
comparing algorithms). Hold out a separate final test set only for reporting
the final, unbiased performance estimate.

---

## Bias-Variance Tradeoff

**What:** Prediction error decomposes into three components: **bias**
(systematic underfitting — the model is too simple to capture the pattern),
**variance** (overfitting — the model memorizes noise in the training data),
and irreducible noise. The tradeoff is that reducing bias typically
increases variance, and vice versa.

**Learning curves interpretation:** Plot training and validation scores
as a function of training set size:
- **High bias (underfitting):** Both curves plateau at a low score with a
  small gap between them. The model needs more capacity (fewer constraints,
  more features, a more complex algorithm).
- **High variance (overfitting):** Training score is high but validation
  score is much lower, with a large gap. The model needs regularization,
  more training data, or a simpler algorithm.
- **Good fit:** Both curves converge to a high score as data increases.

**Example:** A decision tree with no depth limit will achieve ~100% training
accuracy (low bias) but poor test accuracy (high variance). Limiting depth
increases bias slightly but reduces variance significantly, yielding better
generalization.

---

## Ensembles

**What:** Ensemble methods combine multiple base models into a single
predictor. The key insight is that individual models make different errors,
and combining them can cancel out those errors.

**Voting:** Simple majority vote (hard voting) or average predicted
probabilities (soft voting) across all base models. Works well when base
models are diverse (different algorithms or different hyperparameters).

**Stacking:** Train a *meta-learner* (often Logistic Regression) on the
predictions of the base models. The meta-learner learns which base model
to trust for each type of input, typically outperforming simple voting.

**Why they work:** If each base model is correct > 50% of the time and their
errors are independent, the ensemble's accuracy increases as you add more
models (Condorcet's jury theorem). In practice, even with correlated errors,
diversity among base models ensures that the ensemble is more robust than
any single model. This is why competition-winning solutions almost always
use ensembles.

**When to use:** When you need that last few percentage points of accuracy
and can afford the computational cost of training and serving multiple models.
For quick prototypes or latency-sensitive applications, a single well-tuned
model is often preferable.
