"""Stratified k-fold cross-validation evaluator.

Why Cross-Validation Matters:
─────────────────────────────
A single train/test split gives one noisy estimate of model performance.
If you happened to put all the easy examples in the test set, you'd
overestimate accuracy; hard examples would make you underestimate.

**k-fold cross-validation** mitigates this by partitioning the data into
*k* equally-sized folds.  For each of the *k* iterations, one fold is
held out as the validation set while the remaining *k − 1* folds are
used for training.  The metric is computed on each fold and then
averaged, yielding a more robust (lower-variance) performance estimate.

**Stratification** ensures that each fold preserves the same class
distribution as the full dataset.  This is critical for imbalanced
classification problems where a random split might accidentally create
folds missing an entire minority class.

The trade-off: k-fold costs *k×* a single fit.  The standard k = 5 or
k = 10 balances accuracy of the estimate against computational cost.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class CrossValidationEvaluator:
    """Evaluate a classifier with stratified k-fold cross-validation.

    Tracks per-fold metrics (accuracy, precision, recall, F1) and
    aggregates them into a report with mean and standard deviation.

    Attributes:
        n_splits: Number of folds.
        random_state: Seed for reproducibility.
        shuffle: Whether to shuffle before splitting.
        metrics: List of metric keys computed per fold.

    Example::

        >>> from sklearn.linear_model import LogisticRegression
        >>> from src.cross_validation import CrossValidationEvaluator
        >>> evaluator = CrossValidationEvaluator(n_splits=5)
        >>> report = evaluator.evaluate(
        ...     X=X_tfidf, y=y, model_cls=LogisticRegression,
        ...     model_kwargs={"max_iter": 1000},
        ... )
        >>> round(report["mean_accuracy"], 4)
        0.8734
    """

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        shuffle: bool = True,
    ) -> None:
        """Initialise the evaluator.

        Args:
            n_splits: Number of folds. Must be ≥ 2. Defaults to 5.
            random_state: Random seed for reproducibility.
                Defaults to 42.
            shuffle: Whether to shuffle data before splitting.
                Defaults to ``True``.

        Raises:
            ValueError: If *n_splits* < 2.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be ≥ 2, got {n_splits}.")
        self.n_splits: int = n_splits
        self.random_state: int = random_state
        self.shuffle: bool = shuffle
        self.metrics: List[str] = ["accuracy", "precision", "recall", "f1"]
        # Per-fold raw results stored for inspection
        self._fold_results: List[Dict[str, float]] = []

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_cls: Any,
        model_kwargs: Optional[Dict[str, Any]] = None,
        pos_label: int = 1,
        average: str = "weighted",
    ) -> Dict[str, Any]:
        """Run stratified k-fold CV and return a comprehensive report.

        For each fold a fresh model instance is created, trained on the
        training portion, and evaluated on the held-out fold.  Metrics
        are computed on the validation predictions.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.
            model_cls: A scikit-learn-compatible estimator class (must
                expose ``.fit()`` and ``.predict()``).
            model_kwargs: Keyword arguments forwarded to the estimator
                constructor.  Defaults to ``None`` (no extra args).
            pos_label: The positive class label for binary metrics.
                Defaults to 1.
            average: Averaging strategy for multi-class metrics.  One of
                ``"micro"``, ``"macro"``, ``"weighted"``, or ``"binary"``.
                Defaults to ``"weighted"``.

        Returns:
            Dictionary with keys:
            - ``"mean_accuracy"``, ``"mean_precision"``, ``"mean_recall"``,
              ``"mean_f1"``: mean across folds.
            - ``"std_accuracy"``, ``"std_precision"``, ``"std_recall"``,
              ``"std_f1"``: standard deviation across folds.
            - ``"fold_results"``: list of per-fold metric dicts.
            - ``"n_splits"``: number of folds used.
            - ``"elapsed_seconds"``: total wall-clock time.

        Raises:
            ValueError: If *X* and *y* have incompatible shapes.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]}."
            )

        model_kwargs = model_kwargs or {}
        # StratifiedKFold guarantees each fold mirrors the full
        # class distribution — essential for imbalanced datasets.
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        self._fold_results = []
        start_time = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # A fresh model each fold prevents information leakage
            # from previous iterations.
            model = model_cls(**model_kwargs)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Binary metrics default to pos_label when average="binary";
            # "weighted" is generally safer for multi-class.
            acc = float(accuracy_score(y_val, y_pred))
            prec = float(precision_score(y_val, y_pred, pos_label=pos_label, average=average, zero_division=0))
            rec = float(recall_score(y_val, y_pred, pos_label=pos_label, average=average, zero_division=0))
            f1 = float(f1_score(y_val, y_pred, pos_label=pos_label, average=average, zero_division=0))

            fold_report = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
            self._fold_results.append(fold_report)
            logger.info(
                "Fold %d/%d — acc: %.4f  prec: %.4f  rec: %.4f  f1: %.4f",
                fold_idx + 1,
                self.n_splits,
                acc,
                prec,
                rec,
                f1,
            )

        elapsed = time.time() - start_time
        return self._build_report(elapsed)

    def _build_report(self, elapsed: float) -> Dict[str, Any]:
        """Aggregate per-fold results into a summary dict."""
        fold_array = np.array(self._fold_results)  # shape: (n_splits, n_metrics)

        report: Dict[str, Any] = {"n_splits": self.n_splits, "elapsed_seconds": round(elapsed, 2)}

        for i, metric in enumerate(self.metrics):
            values = fold_array[:, i]
            report[f"mean_{metric}"] = round(float(values.mean()), 4)
            report[f"std_{metric}"] = round(float(values.std()), 4)

        report["fold_results"] = self._fold_results
        return report
