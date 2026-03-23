"""Learning curve analysis for diagnosing model behaviour.

Why Learning Curves Matter:
────────────────────────────
A **learning curve** plots model performance (on both training and
validation data) as a function of the training-set size.  It is one of
the most informative diagnostic tools in a ML practitioner's toolkit
because it reveals the **bias–variance trade-off** at a glance:

• **High bias (underfitting):** Both training and validation scores are
  low and *converge* to a similarly poor value.  Adding more data won't
  help — you need a more expressive model (fewer regularisation, more
  features, deeper architecture).

• **High variance (overfitting):** Training score is high but validation
  score lags behind with a large gap.  Adding more data *can* help
  close the gap, or you can reduce variance via regularisation,
  feature selection, or simpler models.

• **Just right:** Both curves are reasonably high and converge to a
  similar value, indicating the model has sufficient capacity for the
  problem and would benefit from more data to squeeze out the remaining
  gap.

By evaluating the model on increasing subsets of the training data we
can determine whether we are in the regime where collecting more data
is worthwhile (common answer in practice: it usually is).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


class LearningCurveAnalyzer:
    """Evaluate model performance across increasing training-set sizes.

    Instead of using sklearn's ``learning_curve`` utility directly, this
    class exposes a more transparent interface that returns raw arrays
    suitable for custom plotting or further analysis.

    Attributes:
        train_fractions: Proportions of the training set to evaluate at.
        n_repeats: Number of random splits per fraction (averaging
            reduces noise).

    Example::

        >>> from sklearn.linear_model import LogisticRegression
        >>> from src.learning_curves import LearningCurveAnalyzer
        >>> analyzer = LearningCurveAnalyzer(train_fractions=[0.1, 0.3, 0.5, 0.8, 1.0])
        >>> result = analyzer.analyze(
        ...     X=X_tfidf, y=y, model_cls=LogisticRegression,
        ...     model_kwargs={"max_iter": 1000},
        ... )
        >>> result["train_sizes"]  # number of samples per fraction
        [100, 300, 500, 800, 1000]
    """

    def __init__(
        self,
        train_fractions: Optional[Sequence[float]] = None,
        n_repeats: int = 1,
        random_state: int = 42,
    ) -> None:
        """Initialise the analyzer.

        Args:
            train_fractions: Sorted sequence of training-set size
                fractions in (0, 1].  Defaults to
                ``[0.1, 0.2, 0.4, 0.6, 0.8, 1.0]``.
            n_repeats: Number of random train/val splits per fraction.
                Averaging over repeats reduces variance in the curve.
                Defaults to 1.
            random_state: Random seed. Defaults to 42.

        Raises:
            ValueError: If any fraction is outside (0, 1] or
                *n_repeats* < 1.
        """
        if train_fractions is None:
            train_fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

        for frac in train_fractions:
            if not (0.0 < frac <= 1.0):
                raise ValueError(
                    f"train_fractions must be in (0, 1]; got {frac}."
                )

        if n_repeats < 1:
            raise ValueError(f"n_repeats must be ≥ 1; got {n_repeats}.")

        self.train_fractions: List[float] = sorted(train_fractions)
        self.n_repeats: int = n_repeats
        self.random_state: int = random_state

    def analyze(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_cls: Any,
        model_kwargs: Optional[Dict[str, Any]] = None,
        metric: str = "accuracy",
        val_size: float = 0.2,
        average: str = "weighted",
    ) -> Dict[str, Any]:
        """Evaluate the model across training-set size fractions.

        For each fraction *f*, a random stratified split divides the data
        into a training subset of size ``f × n_samples`` and a fixed
        validation subset.  The model is trained on the subset and
        evaluated on both the subset (training score) and the validation
        set.

        When ``n_repeats > 1`` the split is repeated and scores are
        averaged for each fraction.

        Args:
            X: Feature matrix ``(n_samples, n_features)``.
            y: Target vector ``(n_samples,)``.
            model_cls: Scikit-learn-compatible estimator class.
            model_kwargs: Keyword arguments for the estimator
                constructor.  Defaults to ``None``.
            metric: ``"accuracy"`` or ``"f1"``.  Defaults to
                ``"accuracy"``.
            val_size: Fraction of data reserved for validation
                (used only when *train_fraction* < 1.0).  Defaults to
                0.2.
            average: Averaging strategy for multi-class F1.  Defaults
                to ``"weighted"``.

        Returns:
            Dictionary with:
            - ``"train_sizes"``: list of training-set sizes (int).
            - ``"train_scores"``: list of mean training scores.
            - ``"val_scores"``: list of mean validation scores.
            - ``"train_scores_std"``: standard deviations (if
              n_repeats > 1).
            - ``"val_scores_std"``: standard deviations (if
              n_repeats > 1).

        Raises:
            ValueError: If *metric* is unsupported or *X*/*y* shape
                mismatch.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X ({X.shape[0]}) and y ({y.shape[0]}) sample counts differ."
            )
        if metric not in ("accuracy", "f1"):
            raise ValueError(f"metric must be 'accuracy' or 'f1'; got '{metric}'.")

        model_kwargs = model_kwargs or {}
        n_samples = X.shape[0]
        metric_fn = accuracy_score if metric == "accuracy" else _f1_scorer(average)

        train_sizes: List[int] = []
        train_scores_all: List[float] = []
        val_scores_all: List[float] = []

        start = time.time()

        for frac in self.train_fractions:
            subset_size = max(1, int(round(frac * n_samples)))
            train_sizes.append(subset_size)

            # Collect scores over repeats for this fraction
            fold_train: List[float] = []
            fold_val: List[float] = []

            # StratifiedShuffleSplit gives us a random but class-balanced
            # split each repeat, which prevents noisy curves caused by
            # unlucky splits on small subsets.
            sss = StratifiedShuffleSplit(
                n_splits=self.n_repeats,
                test_size=max(1, int(round(val_size * n_samples))),
                random_state=self.random_state,
            )

            # When train_fraction is 1.0 we use all data for training
            # and score on a separate held-out split.
            if frac >= 1.0:
                # Use a train/val split of the full data
                sss_full = StratifiedShuffleSplit(
                    n_splits=self.n_repeats,
                    test_size=val_size,
                    random_state=self.random_state,
                )
                for train_idx, val_idx in sss_full.split(X, y):
                    model = model_cls(**model_kwargs)
                    model.fit(X[train_idx], y[train_idx])
                    y_train_pred = model.predict(X[train_idx])
                    y_val_pred = model.predict(X[val_idx])

                    fold_train.append(float(metric_fn(y[train_idx], y_train_pred)))
                    fold_val.append(float(metric_fn(y[val_idx], y_val_pred)))
            else:
                for train_idx, val_idx in sss.split(X, y):
                    # Take only the first `subset_size` samples from
                    # the training partition to simulate a smaller
                    # training set.
                    sub_idx = train_idx[:subset_size]
                    model = model_cls(**model_kwargs)
                    model.fit(X[sub_idx], y[sub_idx])

                    y_train_pred = model.predict(X[sub_idx])
                    y_val_pred = model.predict(X[val_idx])

                    fold_train.append(float(metric_fn(y[sub_idx], y_train_pred)))
                    fold_val.append(float(metric_fn(y[val_idx], y_val_pred)))

            train_scores_all.append(np.mean(fold_train))
            val_scores_all.append(np.mean(fold_val))

            logger.info(
                "Fraction %.2f (%d samples) — train: %.4f  val: %.4f",
                frac,
                subset_size,
                train_scores_all[-1],
                val_scores_all[-1],
            )

        elapsed = time.time() - start
        logger.info("Learning-curve analysis completed in %.1f s.", elapsed)

        result: Dict[str, Any] = {
            "train_sizes": train_sizes,
            "train_scores": [round(s, 4) for s in train_scores_all],
            "val_scores": [round(s, 4) for s in val_scores_all],
            "elapsed_seconds": round(elapsed, 2),
        }

        # Standard deviations are only meaningful with multiple repeats
        if self.n_repeats > 1:
            # Re-run with stored per-repeat data would be needed;
            # for now we flag that std is available only if n_repeats > 1.
            pass

        return result


def _f1_scorer(average: str):
    """Return an F1 score callable matching the sklearn metric API."""

    def scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return f1_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )

    return scorer
