"""TF-IDF feature extraction with interpretability utilities.

TF-IDF (Term Frequency–Inverse Document Frequency) converts raw text into
numerical feature vectors.  Words that appear frequently in a single document
but rarely across the corpus receive high scores, making them discriminative
features for classification.

This module wraps scikit-learn's ``TfidfVectorizer`` and adds a
``get_top_features_per_class()`` method for model interpretability — a
critical skill for portfolio projects where you need to *explain* your
model's decisions, not just report accuracy numbers.
"""

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

    def get_top_features_per_class(
        self,
        model,
        class_names: Optional[List[str]] = None,
        top_n: int = 15,
    ) -> dict:
        """Extract the most discriminative features for each class.

        For linear models (Logistic Regression, Linear SVM), the learned
        coefficients directly indicate feature importance: a high positive
        weight for class *k* means the feature pushes predictions toward
        that class.  This makes the model **interpretable** — you can
        explain *why* a text was classified a certain way by pointing to
        the most influential words.

        This interpretability is a key advantage of TF-IDF + linear
        model pipelines over black-box approaches like deep learning.

        Args:
            model: A fitted linear model with a ``coef_`` attribute
                (e.g., ``LogisticRegression``, ``LinearSVC``).
            class_names: Optional list of human-readable class labels.
                If not provided, integer indices are used.
            top_n: Number of top features to return per class.

        Returns:
            Dictionary mapping class name to a list of
            ``(feature_name, weight)`` tuples sorted by descending
            absolute weight.

        Raises:
            RuntimeError: If the vectorizer has not been fitted.
            AttributeError: If *model* lacks a ``coef_`` attribute.

        Example::

            >>> extractor = TfidfFeatureExtractor()
            >>> X = extractor.fit_transform(texts)
            >>> lr = LogisticRegression().fit(X, labels)
            >>> top = extractor.get_top_features_per_class(lr, top_n=5)
            >>> top["positive"][0]  # ('amazing', 2.134)
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fitted before extracting feature importance.")
        if not hasattr(model, "coef_"):
            raise AttributeError(
                f"{type(model).__name__} has no coef_ attribute. "
                "Feature importance extraction requires a linear model."
            )

        feature_names = self.get_feature_names()
        coefs = model.coef_

        # For binary classification, coef_ is shape (1, n_features);
        # for multiclass it is (n_classes, n_features).
        if coefs.ndim == 1:
            coefs = coefs.reshape(1, -1)

        result = {}
        for i, row in enumerate(coefs):
            label = class_names[i] if class_names and i < len(class_names) else str(i)
            # Sort by absolute weight — the most influential features
            # regardless of direction
            top_indices = np.argsort(np.abs(row))[::-1][:top_n]
            result[label] = [
                (feature_names[idx], round(float(row[idx]), 4))
                for idx in top_indices
            ]
        return result
