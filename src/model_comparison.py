"""Side-by-side model comparison for text classification."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ModelComparison:
    """Compare multiple classifiers on the same dataset.

    Stores per-model metrics and provides convenience methods to
    retrieve the best model by a chosen metric and to render a
    markdown comparison table.

    Example::

        >>> from sklearn.linear_model import LogisticRegression
        >>> comp = ModelComparison()
        >>> comp.add_model("lr", LogisticRegression(), X, y)
        >>> comp.get_best_model("f1")["name"]
        'lr'
    """

    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []

    def add_model(
        self,
        name: str,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Train *model* on *X* / *y* and record metrics.

        Uses :func:`src.evaluation.compute_metrics` for evaluation.

        Args:
            name: Human-readable model identifier.
            model: Scikit-learn compatible estimator with ``fit`` and
                ``predict`` methods.
            X: Feature matrix.
            y: Target labels.

        Returns:
            Dictionary of evaluation metrics for this model.
        """
        from src.evaluation import compute_metrics

        model.fit(X, y)
        y_pred = model.predict(X)

        # Attempt to get probabilities if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X)
            except Exception:
                pass

        metrics = compute_metrics(y, y_pred, y_proba=y_proba)
        metrics["name"] = name

        self.results.append(metrics)
        logger.info("Model '%s' — accuracy: %.4f", name, metrics.get("accuracy", 0))
        return metrics

    def get_results(self) -> List[Dict[str, Any]]:
        """Return all stored model results."""
        return self.results

    def get_best_model(self, metric: str = "f1") -> Dict[str, Any]:
        """Return the result dict for the model with the highest *metric*.

        Args:
            metric: Metric name to rank by (e.g. ``"f1"``, ``"accuracy"``).

        Returns:
            The result dictionary for the best model.

        Raises:
            ValueError: If no models have been added, or *metric* not
                found in any result.
        """
        if not self.results:
            raise ValueError("No models have been added yet.")
        for r in self.results:
            if metric not in r:
                raise ValueError(f"Metric '{metric}' not found in results.")
        return max(self.results, key=lambda r: r.get(metric, 0))

    def to_markdown(self) -> str:
        """Render all results as a markdown table.

        Returns:
            Multi-line markdown string with one column per metric
            and one row per model.
        """
        if not self.results:
            return "| (no models) |"

        # Collect all metric keys (except 'name')
        keys = sorted(
            {k for r in self.results for k in r if k != "name"}
        )

        header = "| name | " + " | ".join(keys) + " |"
        sep = "|" + "------|" * (len(keys) + 1)

        rows = []
        for r in self.results:
            vals = [f"{r.get(k, 'N/A')}" for k in keys]
            rows.append(f"| {r['name']} | " + " | ".join(vals) + " |")

        return "\n".join([header, sep] + rows)
