"""Tests for LearningCurveAnalyzer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from src.learning_curves import LearningCurveAnalyzer


@pytest.fixture
def binary_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42,
    )
    return X, y


class TestLearningCurveAnalyzerInit:
    def test_default_fractions(self):
        analyzer = LearningCurveAnalyzer()
        assert analyzer.train_fractions == [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    def test_custom_fractions_sorted(self):
        analyzer = LearningCurveAnalyzer(train_fractions=[0.8, 0.2, 0.5])
        assert analyzer.train_fractions == [0.2, 0.5, 0.8]

    def test_fraction_zero_raises(self):
        with pytest.raises(ValueError, match="train_fractions must be in"):
            LearningCurveAnalyzer(train_fractions=[0.0])

    def test_fraction_negative_raises(self):
        with pytest.raises(ValueError, match="train_fractions must be in"):
            LearningCurveAnalyzer(train_fractions=[-0.1])

    def test_n_repeats_zero_raises(self):
        with pytest.raises(ValueError, match="n_repeats must be"):
            LearningCurveAnalyzer(n_repeats=0)


class TestLearningCurveAnalyzerAnalyze:
    def test_analyze_returns_expected_keys(self, binary_data):
        X, y = binary_data
        analyzer = LearningCurveAnalyzer(train_fractions=[0.3, 1.0])
        result = analyzer.analyze(X, y, LogisticRegression, {"max_iter": 500})

        assert "train_sizes" in result
        assert "train_scores" in result
        assert "val_scores" in result
        assert "elapsed_seconds" in result

    def test_analyze_train_sizes_increase(self, binary_data):
        X, y = binary_data
        analyzer = LearningCurveAnalyzer(train_fractions=[0.1, 0.5, 1.0])
        result = analyzer.analyze(X, y, LogisticRegression, {"max_iter": 500})

        assert result["train_sizes"] == sorted(result["train_sizes"])
        assert all(s > 0 for s in result["train_sizes"])

    def test_analyze_scores_in_valid_range(self, binary_data):
        X, y = binary_data
        analyzer = LearningCurveAnalyzer(train_fractions=[0.5, 1.0])
        result = analyzer.analyze(X, y, LogisticRegression, {"max_iter": 500})

        for score in result["train_scores"] + result["val_scores"]:
            assert 0.0 <= score <= 1.0

    def test_analyze_invalid_metric_raises(self, binary_data):
        X, y = binary_data
        analyzer = LearningCurveAnalyzer(train_fractions=[0.5])
        with pytest.raises(ValueError, match="metric must be"):
            analyzer.analyze(X, y, LogisticRegression, {"max_iter": 500}, metric="invalid")
