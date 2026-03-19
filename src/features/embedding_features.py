"""Embedding feature extraction (TF-IDF fallback)."""
import logging
from typing import List, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)

HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    logger.info("sentence-transformers not available — using TF-IDF fallback")


class EmbeddingExtractor:
    """Sentence embedding extraction with TF-IDF fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 128) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._model = None
        self._svd: Optional[TruncatedSVD] = None
        self._tfidf: Optional[TfidfVectorizer] = None
        self._is_fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts to embeddings.

        Args:
            texts: List of documents.

        Returns:
            Dense embedding matrix (n_samples, embedding_dim).
        """
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self._model = SentenceTransformer(self.model_name)
                embeddings = self._model.encode(texts, show_progress_bar=False)
                self._is_fitted = True
                logger.info("SentenceTransformer embeddings: %d dims", embeddings.shape[1])
                return embeddings
            except Exception as exc:
                logger.warning("SentenceTransformer failed: %s — using TF-IDF fallback", exc)

        # Fallback: TF-IDF + SVD
        self._tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        tfidf_matrix = self._tfidf.fit_transform(texts)
        self._svd = TruncatedSVD(n_components=self.embedding_dim, random_state=RANDOM_SEED)
        embeddings = self._svd.fit_transform(tfidf_matrix)
        embeddings = normalize(embeddings, norm='l2')
        self._is_fitted = True
        logger.info("TF-IDF+SVD fallback embeddings: %d dims", embeddings.shape[1])
        return embeddings

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted model.

        Args:
            texts: List of documents.

        Returns:
            Embedding matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform first")

        if self._model is not None:
            return self._model.encode(texts, show_progress_bar=False)

        tfidf_matrix = self._tfidf.transform(texts)
        embeddings = self._svd.transform(tfidf_matrix)
        return normalize(embeddings, norm='l2')
