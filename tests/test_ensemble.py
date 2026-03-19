"""Tests for model ensemble."""
import pytest
import numpy as np
from src.models.model_ensemble import ModelEnsemble
from sklearn.linear_model import LogisticRegression


class TestModelEnsemble:
    def test_empty_ensemble_raises(self):
        ens = ModelEnsemble()
        with pytest.raises(RuntimeError, match="No models"):
            ens.predict(np.zeros((5, 10)))

    def test_add_model(self):
        ens = ModelEnsemble()
        model = LogisticRegression()
        ens.add_model("lr", model)
        assert "lr" in ens.model_names

    def test_predict_shape(self):
        ens = ModelEnsemble()
        X = np.random.rand(50, 10)
        y = (X[:, 0] > 0.5).astype(int)
        lr = LogisticRegression(max_iter=100)
        lr.fit(X, y)
        ens.add_model("lr", lr)
        preds = ens.predict(X)
        assert preds.shape == (50,)

    def test_predict_proba_shape(self):
        ens = ModelEnsemble()
        X = np.random.rand(50, 10)
        y = (X[:, 0] > 0.5).astype(int)
        lr = LogisticRegression(max_iter=100)
        lr.fit(X, y)
        ens.add_model("lr", lr)
        proba = ens.predict_proba(X)
        assert proba.shape == (50, 2)

    def test_hard_voting(self):
        ens = ModelEnsemble(strategy="hard")
        X = np.random.rand(50, 10)
        y = (X[:, 0] > 0.5).astype(int)
        lr = LogisticRegression(max_iter=100)
        lr.fit(X, y)
        ens.add_model("lr", lr)
        preds = ens.predict(X)
        assert preds.shape == (50,)
