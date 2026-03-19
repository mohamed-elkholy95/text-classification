"""Evaluation metrics."""
import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        classification_report, precision_recall_curve, roc_curve,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities.

    Returns:
        Metrics dict.
    """
    if not SKLEARN_AVAILABLE:
        return {}

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_proba[:, 1])), 4)
        except (ValueError, IndexError):
            pass
    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Compute confusion matrix.

    Returns:
        Dict with tp, fp, tn, fn counts.
    """
    if not SKLEARN_AVAILABLE:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    cm = confusion_matrix(y_true, y_pred)
    n = min(cm.shape[0], cm.shape[1])
    if n == 2:
        return {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}
    return {"matrix": cm.tolist()}


def generate_evaluation_report(metrics: Dict[str, float]) -> str:
    """Generate markdown report."""
    lines = ["# Text Classification — Evaluation Report", "", "| Metric | Value |", "|--------|-------|"]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
    return "\n".join(lines)
