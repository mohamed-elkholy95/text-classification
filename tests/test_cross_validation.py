"""Tests for CrossValidationEvaluator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from src.cross_validation import CrossValidationEvaluator


@pytest.fixture
def binary_data():
    X, y = make_classification(
        n_samples=120, n_features=10, n_informative=5,
        n_redundant=2, random_state=42,
    )
    return X, y


class TestCrossValidationEvaluatorInit:
    def test_default_params(self):
        evaluator = CrossValidationEvaluator()
        assert evaluator.n_splits == 5
        assert "accuracy" in evaluator.scoring

    def test_custom_n_splits(self):
        evaluator = CrossValidationEvaluator(n_splits=3)
        assert evaluator.n_splits == 3

    def test_custom_scoring(self):
        evaluator = CrossValidationEvaluator(scoring=["accuracy", "precision_weighted", "recall_weighted"])
        assert "precision_weighted" in evaluator.scoring

    def test_n_splits_too_low_raises(self):
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            CrossValidationEvaluator(n_splits=1)


class TestCrossValidationEvaluatorEvaluate:
    def test_evaluate_returns_expected_keys(self, binary_data):
        X, y = binary_data
        evaluator = CrossValidationEvaluator(n_splits=3)
        result = evaluator.evaluate(LogisticRegression(max_iter=500), X, y)

        assert "mean_accuracy" in result
        assert "std_accuracy" in result
        assert "mean_f1_weighted" in result
        assert "std_f1_weighted" in result
        assert "fold_scores" in result

    def test_evaluate_metric_ranges(self, binary_data):
        X, y = binary_data
        evaluator = CrossValidationEvaluator(n_splits=3)
        result = evaluator.evaluate(LogisticRegression(max_iter=500), X, y)

        assert 0.0 <= result["mean_accuracy"] <= 1.0
        assert 0.0 <= result["std_accuracy"] <= 1.0
        assert 0.0 <= result["mean_f1_weighted"] <= 1.0

    def test_evaluate_fold_scores_match_aggregates(self, binary_data):
        X, y = binary_data
        evaluator = CrossValidationEvaluator(n_splits=3)
        result = evaluator.evaluate(LogisticRegression(max_iter=500), X, y)

        fold_accs = result["fold_scores"]["accuracy"]
        assert len(fold_accs) == 3
        assert abs(np.mean(fold_accs) - result["mean_accuracy"]) < 1e-9

    def test_evaluate_custom_scoring(self, binary_data):
        X, y = binary_data
        evaluator = CrossValidationEvaluator(
            n_splits=3,
            scoring=["accuracy", "precision_weighted"],
        )
        result = evaluator.evaluate(LogisticRegression(max_iter=500), X, y)

        assert "mean_precision_weighted" in result
        assert "std_precision_weighted" in result
        assert "precision_weighted" in result["fold_scores"]

    def test_evaluate_reproducible(self, binary_data):
        X, y = binary_data
        evaluator = CrossValidationEvaluator(n_splits=3)
        r1 = evaluator.evaluate(LogisticRegression(max_iter=500), X, y)
        r2 = evaluator.evaluate(LogisticRegression(max_iter=500), X, y)
        assert r1["mean_accuracy"] == r2["mean_accuracy"]
