"""Tests for transformer classifier."""
import pytest
import numpy as np
from src.models.transformer_classifier import TransformerClassifier, HAS_TRANSFORMERS


class TestTransformerClassifier:
    def test_init(self):
        tc = TransformerClassifier(num_labels=2)
        assert tc.num_labels == 2

    def test_predict_unfitted_returns_array(self):
        tc = TransformerClassifier(num_labels=2)
        result = tc.predict(["hello", "world"])
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_predict_proba_unfitted(self):
        tc = TransformerClassifier(num_labels=2)
        result = tc.predict_proba(["hello"])
        assert result.shape == (1, 2)
        assert np.allclose(result.sum(axis=1), 1.0)

    def test_get_config(self):
        tc = TransformerClassifier(model_name="distilbert-base-uncased", num_labels=3)
        config = tc.get_config()
        assert config["model_name"] == "distilbert-base-uncased"
        assert config["num_labels"] == 3
        assert config["is_fitted"] is False
