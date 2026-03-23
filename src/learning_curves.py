"""Learning curve analysis for classification models."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import learning_curve

logger = logging.getLogger(__name__)


class LearningCurveAnalyzer:
    """Generate and analyze learning curves.

    Trains the model on increasing subsets of the training data and
    records training / validation scores to diagnose underfitting,
    overfitting, or data-sufficiency.

    Args:
        cv: Number of cross-validation folds for each subset size.
        num_points: Number of training-set sizes to evaluate.

    Example::

        >>> from sklearn.linear_model import LogisticRegression
        >>> analyzer = LearningCurveAnalyzer(cv=3, num_points=5)
        >>> result = analyzer.analyze(LogisticRegression(), X, y)
        >>> "train_scores" in result
        True
    """

    def __init__(self, cv: int = 5, num_points: int = 10) -> None:
        self.cv = cv
        self.num_points = num_points

    def analyze(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, object]:
        """Compute learning curve data.

        Args:
            model: Scikit-learn compatible estimator.
            X: Feature matrix.
            y: Target labels.

        Returns:
            Dictionary with ``"train_sizes"``, ``"train_scores"``,
            ``"val_scores"``.  Score arrays have shape
            ``(num_sizes, cv)``.
        """
        sizes = np.linspace(
            0.1, 1.0, num=self.num_points, dtype=int,
        )
        # Ensure minimum of 2 samples per fold
        min_samples = max(2, self.cv * 2)
        sizes = np.unique(np.clip(sizes, min_samples, len(y)))

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=sizes,
            cv=self.cv,
            shuffle=True,
            random_state=42,
        )

        return {
            "train_sizes": train_sizes_abs.tolist(),
            "train_scores": train_scores.tolist(),
            "val_scores": val_scores.tolist(),
        }
