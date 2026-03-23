"""Tests for compute_per_class_metrics and generate_confusion_matrix_text."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.evaluation import compute_per_class_metrics, generate_confusion_matrix_text


class TestComputePerClassMetrics:
    def test_binary_classification(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        result = compute_per_class_metrics(y_true, y_pred)

        assert "0" in result
        assert "1" in result
        for cls in ("0", "1"):
            assert "precision" in result[cls]
            assert "recall" in result[cls]
            assert "f1" in result[cls]
            assert "support" in result[cls]

    def test_with_class_names(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        result = compute_per_class_metrics(
            y_true, y_pred, class_names=["negative", "positive"],
        )

        assert "negative" in result
        assert "positive" in result

    def test_support_counts(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1])
        result = compute_per_class_metrics(y_true, y_pred)

        assert result["0"]["support"] == 3
        assert result["1"]["support"] == 2

    def test_metric_ranges(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        result = compute_per_class_metrics(y_true, y_pred)

        for cls in result:
            assert 0.0 <= result[cls]["precision"] <= 1.0
            assert 0.0 <= result[cls]["recall"] <= 1.0
            assert 0.0 <= result[cls]["f1"] <= 1.0

    def test_multiclass(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 0, 2])
        result = compute_per_class_metrics(y_true, y_pred)

        assert len(result) == 3
        for cls in ("0", "1", "2"):
            assert cls in result


class TestGenerateConfusionMatrixText:
    def test_binary_matrix(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        text = generate_confusion_matrix_text(y_true, y_pred)

        assert "pred:" in text
        assert "true:" in text
        # Row 0 (true:0): 2 correct, 0 wrong
        assert "2" in text

    def test_with_class_names(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        text = generate_confusion_matrix_text(
            y_true, y_pred, class_names=["neg", "pos"],
        )

        assert "pred:neg" in text or "pred:pos" in text
        assert "true:neg" in text or "true:pos" in text

    def test_output_is_string(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        text = generate_confusion_matrix_text(y_true, y_pred)
        assert isinstance(text, str)

    def test_multiclass_matrix(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        text = generate_confusion_matrix_text(y_true, y_pred)

        lines = text.strip().split("\n")
        # Header + separator + 3 data rows = 5 lines
        assert len(lines) >= 4
