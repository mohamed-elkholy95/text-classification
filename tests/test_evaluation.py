"""Tests for evaluation metrics."""
import pytest
import numpy as np
from src.evaluation import (
    compute_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
    generate_confusion_matrix_text,
    generate_evaluation_report,
)


class TestComputeMetrics:
    """Tests for the compute_metrics aggregation function."""

    def test_binary(self):
        """Weighted metrics on a simple binary classification."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_with_proba(self):
        """ROC-AUC and PR-AUC should be present when y_proba is given for binary."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        metrics = compute_metrics(y_true, y_pred, y_proba=y_proba)
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics


class TestMulticlassMetrics:
    """Tests for compute_metrics with 3+ classes."""

    def test_three_classes_basic(self):
        """Metrics should be computed with average='weighted' for 3 classes."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 2, 0])
        y_pred = np.array([0, 1, 2, 0, 2, 2, 1, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        # accuracy must be between 0 and 1
        assert 0.0 <= metrics["accuracy"] <= 1.0
        # ROC-AUC should NOT be present for multiclass (only binary)
        assert "roc_auc" not in metrics

    def test_three_classes_with_proba_no_roc(self):
        """Even with y_proba provided, ROC-AUC is skipped for multiclass."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        y_proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        metrics = compute_metrics(y_true, y_pred, y_proba=y_proba)
        assert "roc_auc" not in metrics

    def test_five_classes(self):
        """Metrics should work correctly with 5 classes."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 5, size=50)
        y_pred = y_true.copy()
        rng.shuffle(y_pred)
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] >= 0.0
        assert metrics["f1"] >= 0.0


class TestPerClassMetrics:
    """Tests for compute_per_class_metrics — per-class precision/recall/F1."""

    def test_binary_per_class(self):
        """Binary data should produce metrics for both classes."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        result = compute_per_class_metrics(y_true, y_pred)
        assert "0" in result
        assert "1" in result
        for cls_key in ("0", "1"):
            assert "precision" in result[cls_key]
            assert "recall" in result[cls_key]
            assert "f1" in result[cls_key]
            assert "support" in result[cls_key]
        # Class 1 has support=3
        assert result["1"]["support"] == 3
        # Class 0 has support=2
        assert result["0"]["support"] == 2

    def test_binary_with_class_names(self):
        """Class names should be used as dictionary keys when provided."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = compute_per_class_metrics(y_true, y_pred, class_names=["neg", "pos"])
        assert "neg" in result
        assert "pos" in result
        assert result["neg"]["precision"] == 1.0
        assert result["pos"]["precision"] == 1.0

    def test_multiclass_per_class_fallback_to_binary(self):
        """compute_per_class_metrics uses binary-average internally.

        For multiclass data this currently raises a ValueError because
        sklearn's precision_score with average='binary' and more than 2
        present classes is unsupported.  This test documents the current
        limitation so a future fix can flip it to assert success.
        """
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])
        with pytest.raises(ValueError, match="average='binary'"):
            compute_per_class_metrics(y_true, y_pred)

    def test_multiclass_with_class_names(self):
        """Same binary-average limitation applies with class_names."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="average='binary'"):
            compute_per_class_metrics(
                y_true, y_pred, class_names=["pos", "neu", "neg"],
            )


class TestConfusionMatrixText:
    """Tests for generate_confusion_matrix_text formatting."""

    def test_binary_format(self):
        """Binary matrix should have correct structure and counts."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1])
        text = generate_confusion_matrix_text(y_true, y_pred)
        assert "|" in text
        assert "true" in text.lower()
        assert "pred" in text.lower()

    def test_binary_with_class_names(self):
        """Class names should appear in the header row."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        text = generate_confusion_matrix_text(y_true, y_pred, class_names=["neg", "pos"])
        assert "pred:neg" in text
        assert "pred:pos" in text
        assert "true:neg" in text
        assert "true:pos" in text

    def test_multiclass_format(self):
        """3x3 matrix should have header + separator + 3 data rows."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])
        text = generate_confusion_matrix_text(y_true, y_pred)
        lines = text.strip().split("\n")
        # Header + separator + 3 rows = 5 lines
        assert len(lines) == 5

    def test_returns_string(self):
        """Should always return a string."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        result = generate_confusion_matrix_text(y_true, y_pred)
        assert isinstance(result, str)


class TestEdgeCases:
    """Edge-case scenarios for evaluation functions."""

    def test_perfect_predictions(self):
        """All predictions correct — all metrics should be 1.0."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

        # Per-class metrics should all be perfect
        pcm = compute_per_class_metrics(y_true, y_pred)
        for cls_metrics in pcm.values():
            assert cls_metrics["precision"] == 1.0
            assert cls_metrics["recall"] == 1.0
            assert cls_metrics["f1"] == 1.0

        # Confusion matrix should be identity
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm["tn"] == 3
        assert cm["tp"] == 3
        assert cm["fp"] == 0
        assert cm["fn"] == 0

    def test_all_wrong_predictions(self):
        """Every prediction wrong — accuracy should be 0.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

        # Confusion matrix: everything is off-diagonal
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm["tn"] == 0
        assert cm["tp"] == 0
        assert cm["fp"] == 3
        assert cm["fn"] == 3

    def test_single_class(self):
        """When all labels are the same class, metrics should still compute.

        Note: compute_confusion_matrix returns {"matrix": [[4]]} when only
        one class is present (n < 2), because the binary tn/fp/fn/tp
        decomposition requires exactly 2 classes.
        """
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        # With only one class, precision/recall use weighted average —
        # zero_division=0 ensures no NaN.
        assert metrics["accuracy"] == 1.0

        # Confusion matrix falls back to full matrix for non-binary case
        cm = compute_confusion_matrix(y_true, y_pred)
        assert "matrix" in cm
        assert cm["matrix"] == [[4]]

    def test_single_class_all_wrong(self):
        """Single class ground truth but all predictions are the other class.

        With 2 distinct labels present (0 and 1), this is actually a
        valid binary case for the confusion matrix.
        """
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

        # Now it's binary (both 0 and 1 present)
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm["tn"] == 0
        assert cm["fp"] == 3

    def test_per_class_single_class(self):
        """Per-class metrics with only one class should return one entry."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        result = compute_per_class_metrics(y_true, y_pred)
        assert "0" in result
        assert len(result) == 1

    def test_multiclass_all_wrong(self):
        """Multiclass: every prediction is wrong."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 2, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0


class TestConfusionMatrix:
    def test_binary(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert "tp" in cm
        assert "fp" in cm
        assert "tn" in cm
        assert "fn" in cm
        assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == 5


class TestReport:
    def test_generates_markdown(self):
        metrics = {"accuracy": 0.9, "f1": 0.85}
        report = generate_evaluation_report(metrics)
        assert "# Text Classification" in report
        assert "0.9" in report
