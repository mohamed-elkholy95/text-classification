"""Tests for model persistence (save/load/metadata).

Verifying that a model can be saved and loaded with identical
predictions is essential for deployment confidence — you need to
know that serialization preserves the model exactly.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.persistence import (
    list_saved_models,
    load_metadata,
    load_model,
    save_model,
)


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Provide a temporary directory for model artifacts."""
    return tmp_path / "models"


@pytest.fixture
def fitted_model():
    """Return a small fitted LogisticRegression for testing."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    return model, X


class TestSaveModel:
    """Tests for save_model()."""

    def test_save_creates_joblib_file(self, fitted_model, tmp_model_dir):
        model, _ = fitted_model
        path = save_model(model, "test_lr", directory=tmp_model_dir)
        assert path.exists()
        assert path.suffix == ".joblib"

    def test_save_creates_metadata_file(self, fitted_model, tmp_model_dir):
        model, _ = fitted_model
        save_model(model, "test_lr", directory=tmp_model_dir)
        meta_path = tmp_model_dir / "test_lr.meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["model_class"] == "LogisticRegression"
        assert "saved_at" in meta

    def test_save_includes_custom_metadata(self, fitted_model, tmp_model_dir):
        model, _ = fitted_model
        save_model(model, "test_lr", metadata={"C": 1.0, "accuracy": 0.95}, directory=tmp_model_dir)
        meta = json.loads((tmp_model_dir / "test_lr.meta.json").read_text())
        assert meta["custom"]["C"] == 1.0
        assert meta["custom"]["accuracy"] == 0.95

    def test_save_empty_name_raises(self, fitted_model, tmp_model_dir):
        model, _ = fitted_model
        with pytest.raises(ValueError, match="name"):
            save_model(model, "", directory=tmp_model_dir)


class TestLoadModel:
    """Tests for load_model()."""

    def test_round_trip_preserves_predictions(self, fitted_model, tmp_model_dir):
        """Saved and loaded model must produce identical predictions."""
        model, X = fitted_model
        original_preds = model.predict(X)
        save_model(model, "round_trip", directory=tmp_model_dir)
        loaded = load_model("round_trip", directory=tmp_model_dir)
        loaded_preds = loaded.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_load_nonexistent_raises(self, tmp_model_dir):
        with pytest.raises(FileNotFoundError, match="No saved model"):
            load_model("nonexistent", directory=tmp_model_dir)


class TestListModels:
    """Tests for list_saved_models()."""

    def test_list_empty_directory(self, tmp_path):
        assert list_saved_models(tmp_path) == []

    def test_list_returns_saved_names(self, fitted_model, tmp_model_dir):
        model, _ = fitted_model
        save_model(model, "alpha", directory=tmp_model_dir)
        save_model(model, "beta", directory=tmp_model_dir)
        names = list_saved_models(tmp_model_dir)
        assert names == ["alpha", "beta"]


class TestLoadMetadata:
    """Tests for load_metadata()."""

    def test_load_metadata_round_trip(self, fitted_model, tmp_model_dir):
        model, _ = fitted_model
        save_model(model, "meta_test", metadata={"version": "1.0"}, directory=tmp_model_dir)
        meta = load_metadata("meta_test", directory=tmp_model_dir)
        assert meta["custom"]["version"] == "1.0"

    def test_load_missing_metadata_raises(self, tmp_model_dir):
        with pytest.raises(FileNotFoundError):
            load_metadata("nonexistent", directory=tmp_model_dir)
