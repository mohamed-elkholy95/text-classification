"""Tests for TF-IDF features."""
import pytest
import numpy as np
from src.features.tfidf_features import TfidfFeatureExtractor


class TestTfidfFeatureExtractor:
    def test_fit_transform(self, tfidf_matrix):
        assert tfidf_matrix.shape[0] > 0

    def test_transform_after_fit(self, tfidf_extractor, cleaned_data):
        # Fit first
        tfidf_extractor.fit_transform(cleaned_data["text"].tolist())
        # Then transform new texts
        texts = cleaned_data["text"].tolist()[:5]
        X = tfidf_extractor.transform(texts)
        assert X.shape[0] == 5

    def test_transform_before_fit_raises(self, tfidf_extractor):
        with pytest.raises(RuntimeError, match="fit_transform"):
            tfidf_extractor.transform(["hello world"])

    def test_get_feature_names(self):
        extractor = TfidfFeatureExtractor(max_features=100, min_df=1)
        extractor.fit_transform(["hello world foo bar", "foo bar baz qux", "hello baz qux"])
        names = extractor.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_feature_names_unfitted(self, tfidf_extractor):
        assert tfidf_extractor.get_feature_names() == []
