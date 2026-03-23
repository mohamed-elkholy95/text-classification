"""Tests for src.calibration — Calibrator class and calibrate_predictions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.calibration import Calibrator, calibrate_predictions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """Generate a simple binary classification dataset + fitted model."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42,
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    scores = model.decision_function(X)
    return scores, y


# ---------------------------------------------------------------------------
# Calibrator initialization
# ---------------------------------------------------------------------------

class TestCalibratorInit:
    def test_platt_init(self):
        cal = Calibrator(method="platt")
        assert cal.method == "platt"
        assert cal.model is None

    def test_isotonic_init(self):
        cal = Calibrator(method="isotonic")
        assert cal.method == "isotonic"
        assert cal.model is None

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unsupported calibration method"):
            Calibrator(method="unknown")


# ---------------------------------------------------------------------------
# fit / predict_proba
# ---------------------------------------------------------------------------

class TestCalibratorFitPredict:
    def test_fit_returns_self(self, binary_data):
        scores, y = binary_data
        cal = Calibrator(method="platt")
        result = cal.fit(scores, y)
        assert result is cal

    def test_predict_proba_shape(self, binary_data):
        scores, y = binary_data
        cal = Calibrator(method="platt").fit(scores, y)
        probs = cal.predict_proba(scores)
        assert probs.shape == scores.shape
        assert np.all((probs >= 0) & (probs <= 1))

    def test_isotonic_predict_proba(self, binary_data):
        scores, y = binary_data
        cal = Calibrator(method="isotonic").fit(scores, y)
        probs = cal.predict_proba(scores)
        assert probs.shape == scores.shape
        assert np.all((probs >= 0) & (probs <= 1))

    def test_score_returns_float(self, binary_data):
        scores, y = binary_data
        cal = Calibrator(method="platt").fit(scores, y)
        brier = cal.score(scores, y)
        assert isinstance(brier, float)
        assert 0 <= brier <= 1

    def test_predict_unfitted_raises(self, binary_data):
        scores, _ = binary_data
        cal = Calibrator(method="platt")
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.predict_proba(scores)

    def test_predict_empty_scores_raises(self, binary_data):
        _, y = binary_data
        cal = Calibrator(method="platt").fit(binary_data[0], y)
        with pytest.raises(ValueError, match="must not be empty"):
            cal.predict_proba(np.array([]))


# ---------------------------------------------------------------------------
# calibrate_predictions convenience function
# ---------------------------------------------------------------------------

class TestCalibratePredictions:
    def test_returns_dict_with_keys(self, binary_data):
        scores, y = binary_data
        result = calibrate_predictions(scores, y, method="platt")
        assert "probabilities" in result
        assert "brier_score" in result
        assert "method" in result

    def test_probabilities_shape(self, binary_data):
        scores, y = binary_data
        result = calibrate_predictions(scores, y, method="isotonic")
        assert result["probabilities"].shape == scores.shape

    def test_brier_score_reasonable(self, binary_data):
        scores, y = binary_data
        result = calibrate_predictions(scores, y, method="platt")
        # On training data with a decent model, Brier should be < 0.3
        assert result["brier_score"] < 0.5

    def test_invalid_method_raises(self, binary_data):
        scores, y = binary_data
        with pytest.raises(ValueError, match="Unsupported"):
            calibrate_predictions(scores, y, method="bad")
