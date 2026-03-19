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
