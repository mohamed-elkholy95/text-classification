"""Tests for embedding features."""
import pytest
import numpy as np
from scipy.sparse import csr_matrix
from src.features.embedding_features import EmbeddingExtractor
from src.features.feature_combiner import FeatureCombiner


class TestEmbeddingExtractor:
    def test_fit_transform(self):
        ext = EmbeddingExtractor(embedding_dim=2)
        texts = ["hello world this is a test of embeddings"] * 10 + \
                ["foo bar baz qux testing more words"] * 10 + \
                ["completely different sentence here now"] * 10
        emb = ext.fit_transform(texts)
        assert emb.shape[0] == 30
        assert emb.shape[1] == 2

    def test_transform_after_fit(self):
        ext = EmbeddingExtractor(embedding_dim=2)
        train_texts = ["hello world test embeddings"] * 10 + \
                      ["foo bar baz different words"] * 10
        ext.fit_transform(train_texts)
        new_emb = ext.transform(["new text"])
        assert new_emb.shape[0] == 1

    def test_transform_before_fit_raises(self):
        ext = EmbeddingExtractor(embedding_dim=32)
        with pytest.raises(RuntimeError, match="fit_transform"):
            ext.transform(["hello"])


class TestFeatureCombiner:
    def test_sparse_only(self):
        sparse = csr_matrix(np.random.rand(5, 10))
        result = FeatureCombiner.combine(sparse)
        assert result.shape == (5, 10)

    def test_with_embeddings(self):
        sparse = csr_matrix(np.random.rand(5, 10))
        dense = np.random.rand(5, 32)
        result = FeatureCombiner.combine(sparse, dense)
        assert result.shape == (5, 42)
