"""Tests for baseline models."""
import pytest
import numpy as np
from src.models.baseline_models import (
    train_naive_bayes, train_logistic_regression,
    train_svm, train_random_forest, train_all_baselines,
)


class TestTrainNaiveBayes:
    def test_returns_model(self, tfidf_matrix, sample_labels_long):
        model = train_naive_bayes(tfidf_matrix, sample_labels_long)
        assert model is not None

    def test_predict_shape(self, tfidf_matrix, sample_labels_long):
        model = train_naive_bayes(tfidf_matrix, sample_labels_long)
        preds = model.predict(tfidf_matrix)
        assert preds.shape == sample_labels_long.shape


class TestTrainLogisticRegression:
    def test_returns_model(self, tfidf_matrix, sample_labels_long):
        model = train_logistic_regression(tfidf_matrix, sample_labels_long)
        assert model is not None

    def test_predict_shape(self, tfidf_matrix, sample_labels_long):
        model = train_logistic_regression(tfidf_matrix, sample_labels_long)
        preds = model.predict(tfidf_matrix)
        assert preds.shape == sample_labels_long.shape


class TestTrainSVM:
    def test_returns_model(self, tfidf_matrix, sample_labels_long):
        model = train_svm(tfidf_matrix, sample_labels_long)
        assert model is not None

    def test_predict_shape(self, tfidf_matrix, sample_labels_long):
        model = train_svm(tfidf_matrix, sample_labels_long)
        preds = model.predict(tfidf_matrix)
        assert preds.shape == sample_labels_long.shape


class TestTrainRandomForest:
    def test_returns_model(self, tfidf_matrix, sample_labels_long):
        model = train_random_forest(tfidf_matrix, sample_labels_long)
        assert model is not None

    def test_predict_shape(self, tfidf_matrix, sample_labels_long):
        model = train_random_forest(tfidf_matrix, sample_labels_long)
        preds = model.predict(tfidf_matrix)
        assert preds.shape == sample_labels_long.shape


class TestTrainAllBaselines:
    def test_returns_four_models(self, tfidf_matrix, sample_labels_long):
        models = train_all_baselines(tfidf_matrix, sample_labels_long)
        assert len(models) == 4

    def test_model_names(self, tfidf_matrix, sample_labels_long):
        models = train_all_baselines(tfidf_matrix, sample_labels_long)
        assert "naive_bayes" in models
        assert "logistic_regression" in models
        assert "svm" in models
        assert "random_forest" in models
