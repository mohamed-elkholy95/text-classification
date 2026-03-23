"""Cross-validation evaluation for classification models."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

logger = logging.getLogger(__name__)


class CrossValidationEvaluator:
    """Stratified k-fold cross-validation wrapper.

    Runs stratified k-fold CV and aggregates metrics (mean ± std)
    across folds for robust model evaluation.

    Args:
        n_splits: Number of CV folds.  Defaults to 5.
        scoring: Scoring metrics (passed to
            :func:`sklearn.model_selection.cross_validate`).

    Example::

        >>> from sklearn.linear_model import LogisticRegression
        >>> evaluator = CrossValidationEvaluator(n_splits=3)
        >>> result = evaluator.evaluate(LogisticRegression(), X, y)
        >>> result["mean_accuracy"] > 0.5
        True
    """

    def __init__(
        self,
        n_splits: int = 5,
        scoring: Optional[List[str]] = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2.")
        self.n_splits = n_splits
        self.scoring = scoring or ["accuracy", "f1_weighted"]

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Run stratified k-fold CV and return aggregated metrics.

        Args:
            model: Scikit-learn compatible estimator.
            X: Feature matrix.
            y: Target labels.

        Returns:
            Dictionary with ``mean_<metric>`` and ``std_<metric>``
            for every scoring key, plus ``"fold_scores"``.
        """
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        results = cross_validate(model, X, y, cv=cv, scoring=self.scoring)

        aggregated: Dict[str, float] = {}
        for metric in self.scoring:
            key = f"test_{metric}"
            values = results[key]
            aggregated[f"mean_{metric}"] = float(np.mean(values))
            aggregated[f"std_{metric}"] = float(np.std(values))
        aggregated["fold_scores"] = {
            metric: results[f"test_{metric}"].tolist()
            for metric in self.scoring
        }
        return aggregated
