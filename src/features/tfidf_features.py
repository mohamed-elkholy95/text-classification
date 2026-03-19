"""TF-IDF feature extraction."""
import logging
from typing import List, Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TFIDF_CONFIG

logger = logging.getLogger(__name__)


class TfidfFeatureExtractor:
    """TF-IDF vectorizer with configurable n-grams."""

    def __init__(self, **kwargs) -> None:
        config = {**TFIDF_CONFIG, **kwargs}
        self._vectorizer = TfidfVectorizer(
            ngram_range=config.get("ngram_range", (1, 2)),
            max_features=config.get("max_features", 20000),
            min_df=config.get("min_df", 2),
            max_df=config.get("max_df", 0.95),
            sublinear_tf=True,
        )
        self._is_fitted = False

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """Fit vectorizer and transform texts.

        Args:
            texts: List of text documents.

        Returns:
            Sparse TF-IDF matrix.
        """
        X = self._vectorizer.fit_transform(texts)
        self._is_fitted = True
        logger.info("TF-IDF fitted: %d features", X.shape[1])
        return X

    def transform(self, texts: List[str]) -> csr_matrix:
        """Transform texts using fitted vectorizer.

        Args:
            texts: List of text documents.

        Returns:
            Sparse TF-IDF matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform first")
        return self._vectorizer.transform(texts)

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if not self._is_fitted:
            return []
        return self._vectorizer.get_feature_names_out().tolist()
