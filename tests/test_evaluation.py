"""Tests for evaluation metrics."""
import pytest
import numpy as np
from src.evaluation import compute_metrics, compute_confusion_matrix, generate_evaluation_report


class TestComputeMetrics:
    def test_binary(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_with_proba(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        metrics = compute_metrics(y_true, y_pred, y_proba)
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics


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
