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


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, and F1 for each class independently.

    Unlike :func:`compute_metrics` which aggregates with ``average="weighted"``,
    this returns a separate dict per class so you can spot per-class
    failure modes (e.g. one class dragging down the macro average).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Optional list of class name strings.  If provided,
            used as dictionary keys instead of integer class labels.

    Returns:
        Dictionary mapping each class label (int or str from *class_names*)
        to a dict with ``precision``, ``recall``, ``f1``, ``support``.
    """
    if not SKLEARN_AVAILABLE:
        return {}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    result: Dict[str, Dict[str, float]] = {}
    for i, lbl in enumerate(labels):
        key = str(class_names[i]) if class_names and i < len(class_names) else str(lbl)
        result[key] = {
            "precision": round(float(precision_score(
                y_true, y_pred, labels=[lbl], pos_label=lbl, zero_division=0,
            )), 4),
            "recall": round(float(recall_score(
                y_true, y_pred, labels=[lbl], pos_label=lbl, zero_division=0,
            )), 4),
            "f1": round(float(f1_score(
                y_true, y_pred, labels=[lbl], pos_label=lbl, zero_division=0,
            )), 4),
            "support": int((y_true == lbl).sum()),
        }
    return result


def generate_confusion_matrix_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
) -> str:
    """Render a confusion matrix as a human-readable text table.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Optional display names for each class.

    Returns:
        Multi-line string with header row and counts.

    Example::

        >>> print(generate_confusion_matrix_text([0,1,0,1], [0,0,0,1],
        ...                                        class_names=["neg","pos"]))
        |          |  pred:neg |  pred:pos |
        |----------|-----------|-----------|
        | true:neg |         2 |         1 |
        | true:pos |         1 |         1 |
    """
    if not SKLEARN_AVAILABLE:
        return "scikit-learn not available"

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    if class_names and len(class_names) >= len(labels):
        display = [str(class_names[labels.index(l)]) for l in labels]
    else:
        display = [str(l) for l in labels]

    n = len(labels)
    # Compute column widths
    pred_headers = [f"pred:{d}" for d in display]
    row_headers = [f"true:{d}" for d in display]
    col_w = max(len(h) for h in pred_headers) + 2
    row_w = max(len(h) for h in row_headers) + 2

    lines: list[str] = []

    # Header row 1
    header = f"| {'':^{row_w}} | " + " | ".join(f"{h:^{col_w}}" for h in pred_headers) + " |"
    lines.append(header)

    # Separator
    sep = "|" + "-" * (row_w + 2) + "|" + "|".join("-" * (col_w + 2) for _ in range(n)) + "|"
    lines.append(sep)

    # Data rows
    for i in range(n):
        row = f"| {row_headers[i]:^{row_w}} | " + " | ".join(
            f"{int(cm[i, j]):^{col_w}}" for j in range(n)
        ) + " |"
        lines.append(row)

    return "\n".join(lines)
