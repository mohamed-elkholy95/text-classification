"""Tests for the end-to-end TextClassificationPipeline."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.data.preprocessor import TextPreprocessor
from src.features.tfidf_features import TfidfFeatureExtractor
from src.pipeline import PredictionResult, TextClassificationPipeline


@pytest.fixture
def trained_pipeline():
    """Build a minimal trained pipeline for testing."""
    texts = [
        "this is a great product I love it",
        "amazing wonderful fantastic service",
        "terrible awful horrible experience",
        "worst product ever do not buy this",
        "really good quality and fast delivery",
        "bad quality broke after one day",
    ] * 5  # Repeat to have enough samples for TF-IDF
    labels = np.array([1, 1, 0, 0, 1, 0] * 5)

    preprocessor = TextPreprocessor()
    extractor = TfidfFeatureExtractor(max_features=200, min_df=1)
    X = extractor.fit_transform([preprocessor.clean_text(t) for t in texts])

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, labels)

    pipeline = TextClassificationPipeline(
        preprocessor=preprocessor,
        feature_extractor=extractor,
        model=model,
        class_names=["negative", "positive"],
    )
    return pipeline


class TestPredictionResult:
    """Tests for the PredictionResult dataclass."""

    def test_fields_are_accessible(self):
        result = PredictionResult(
            text="hello", label=1, class_name="positive",
            confidence=0.95, probabilities={"negative": 0.05, "positive": 0.95},
        )
        assert result.text == "hello"
        assert result.label == 1
        assert result.confidence == 0.95

    def test_default_probabilities(self):
        result = PredictionResult(text="x", label=0, class_name="neg", confidence=0.5)
        assert result.probabilities == {}


class TestPipelinePredict:
    """Tests for single-text prediction."""

    def test_returns_prediction_result(self, trained_pipeline):
        result = trained_pipeline.predict("This product is amazing")
        assert isinstance(result, PredictionResult)

    def test_label_is_integer(self, trained_pipeline):
        result = trained_pipeline.predict("terrible experience")
        assert isinstance(result.label, int)
        assert result.label in (0, 1)

    def test_confidence_in_valid_range(self, trained_pipeline):
        result = trained_pipeline.predict("really good stuff")
        assert 0.0 <= result.confidence <= 1.0

    def test_probabilities_sum_to_one(self, trained_pipeline):
        result = trained_pipeline.predict("some random text")
        if result.probabilities:
            total = sum(result.probabilities.values())
            assert abs(total - 1.0) < 0.01

    def test_class_name_matches_label(self, trained_pipeline):
        result = trained_pipeline.predict("this is wonderful")
        expected_names = {0: "negative", 1: "positive"}
        assert result.class_name == expected_names[result.label]

    def test_handles_empty_after_cleaning(self, trained_pipeline):
        """Text that becomes empty after cleaning should not crash."""
        result = trained_pipeline.predict("@#$%^&*()")
        assert isinstance(result, PredictionResult)


class TestPipelineBatchPredict:
    """Tests for batch prediction."""

    def test_batch_returns_correct_count(self, trained_pipeline):
        texts = ["good product", "bad product", "okay product"]
        results = trained_pipeline.predict_batch(texts)
        assert len(results) == 3

    def test_batch_preserves_order(self, trained_pipeline):
        texts = ["first text", "second text"]
        results = trained_pipeline.predict_batch(texts)
        assert results[0].text == "first text"
        assert results[1].text == "second text"

    def test_empty_batch_returns_empty_list(self, trained_pipeline):
        assert trained_pipeline.predict_batch([]) == []

    def test_batch_results_are_prediction_results(self, trained_pipeline):
        results = trained_pipeline.predict_batch(["hello world"])
        assert all(isinstance(r, PredictionResult) for r in results)
