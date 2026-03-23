"""Tests for ModelComparison."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification

from src.model_comparison import ModelComparison


@pytest.fixture
def binary_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42,
    )
    return X, y


class TestModelComparisonInit:
    def test_init_empty(self):
        comp = ModelComparison()
        assert comp.get_results() == []

    def test_get_results_empty(self):
        comp = ModelComparison()
        assert comp.get_results() == []


class TestModelComparisonAddModel:
    def test_add_model_returns_metrics(self, binary_data):
        X, y = binary_data
        comp = ModelComparison()
        metrics = comp.add_model("lr", LogisticRegression(max_iter=500), X, y)

        assert isinstance(metrics, dict)
        assert "name" in metrics
        assert metrics["name"] == "lr"
        assert "accuracy" in metrics

    def test_add_multiple_models(self, binary_data):
        X, y = binary_data
        comp = ModelComparison()
        comp.add_model("lr", LogisticRegression(max_iter=500), X, y)
        comp.add_model("dummy", DummyClassifier(strategy="most_frequent"), X, y)

        assert len(comp.get_results()) == 2

    def test_metrics_in_valid_range(self, binary_data):
        X, y = binary_data
        comp = ModelComparison()
        metrics = comp.add_model("lr", LogisticRegression(max_iter=500), X, y)

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0


class TestModelComparisonGetBestModel:
    def test_best_model_no_models_raises(self):
        comp = ModelComparison()
        with pytest.raises(ValueError, match="No models have been added"):
            comp.get_best_model()

    def test_best_model_by_f1(self, binary_data):
        X, y = binary_data
        comp = ModelComparison()
        comp.add_model("lr", LogisticRegression(max_iter=500), X, y)
        comp.add_model("dummy", DummyClassifier(strategy="most_frequent"), X, y)

        best = comp.get_best_model("f1")
        assert best["name"] == "lr"

    def test_best_model_by_accuracy(self, binary_data):
        X, y = binary_data
        comp = ModelComparison()
        comp.add_model("lr", LogisticRegression(max_iter=500), X, y)
        comp.add_model("dummy", DummyClassifier(strategy="most_frequent"), X, y)

        best = comp.get_best_model("accuracy")
        assert best["name"] == "lr"

    def test_best_model_invalid_metric_raises(self, binary_data):
        X, y = binary_data
        comp = ModelComparison()
        comp.add_model("lr", LogisticRegression(max_iter=500), X, y)

        with pytest.raises(ValueError, match="not found"):
            comp.get_best_model("nonexistent_metric")


class TestModelComparisonToMarkdown:
    def test_to_markdown_empty(self):
        comp = ModelComparison()
        md = comp.to_markdown()
        assert "no models" in md

    def test_to_markdown_with_models(self, binary_data):
        X, y = binary_data
        comp = ModelComparison()
        comp.add_model("lr", LogisticRegression(max_iter=500), X, y)
        md = comp.to_markdown()

        assert "lr" in md
        assert "|" in md  # markdown table format
