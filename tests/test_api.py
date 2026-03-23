"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


class TestHealthEndpoint:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    def test_predict(self):
        resp = client.post("/predict", json={"text": "This is great"})
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_label" in data
        assert "confidence" in data

    def test_predict_empty_raises(self):
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict(self):
        resp = client.post("/batch_predict", json={"texts": ["Great!", "Terrible"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2


class TestStatsEndpoint:
    """Tests for the /stats text analysis endpoint.

    The /stats endpoint computes linguistic statistics (word count,
    sentence count, averages, vocabulary richness, readability) inline
    from the input text — no model loading required.
    """

    def test_stats_basic(self):
        """Verify /stats returns all expected fields for a simple sentence."""
        resp = client.post("/stats", json={"text": "The quick brown fox jumps over the lazy dog."})
        assert resp.status_code == 200
        data = resp.json()
        # All expected keys must be present
        for key in ("word_count", "sentence_count", "avg_word_length",
                     "avg_sentence_length", "vocabulary_richness", "readability_score"):
            assert key in data, f"Missing key: {key}"
        assert data["word_count"] == 9
        assert data["sentence_count"] == 1

    def test_stats_multiple_sentences(self):
        """Multiple sentences should produce sentence_count > 1."""
        resp = client.post("/stats", json={"text": "Hello world. How are you? I am fine!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentence_count"] == 3
        # Vocabulary richness = unique words / total words
        assert 0 < data["vocabulary_richness"] <= 1

    def test_stats_vocabulary_richness(self):
        """Text with repeated words should have lower richness than unique text."""
        resp_rep = client.post("/stats", json={"text": "the the the the the the the the"})
        resp_uni = client.post("/stats", json={"text": "the quick brown fox jumps"})
        assert resp_rep.status_code == 200
        assert resp_uni.status_code == 200
        # Repeated text: richness = 1/8 = 0.125, unique: richness = 5/5 = 1.0
        assert resp_rep.json()["vocabulary_richness"] < resp_uni.json()["vocabulary_richness"]

    def test_stats_empty_text(self):
        """Empty text should still return 200 with zeroed stats."""
        resp = client.post("/stats", json={"text": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data["word_count"] == 0


class TestCompareEndpoint:
    """Tests for the /compare model comparison endpoint.

    /compare returns a summary of model performances so the caller
    can compare classifiers without training them locally.
    """

    def test_compare_returns_models(self):
        """Response must include a list of model results and a best_model name."""
        resp = client.post("/compare", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "best_model" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0

    def test_compare_models_have_metrics(self):
        """Each model entry should contain at least accuracy and f1_score keys."""
        resp = client.post("/compare", json={})
        data = resp.json()
        for model in data["models"]:
            assert "name" in model
            assert "accuracy" in model
            assert "f1_score" in model

    def test_compare_best_model_in_list(self):
        """best_model should reference one of the model names in the list."""
        resp = client.post("/compare", json={})
        data = resp.json()
        names = [m["name"] for m in data["models"]]
        assert data["best_model"] in names
