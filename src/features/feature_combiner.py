"""Feature combination."""
import logging
from typing import Optional

import numpy as np
from scipy.sparse import issparse, hstack

logger = logging.getLogger(__name__)


class FeatureCombiner:
    """Combine sparse and dense feature matrices."""

    @staticmethod
    def combine(
        tfidf_matrix,
        embedding_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Combine TF-IDF (sparse) and embedding (dense) features.

        Args:
            tfidf_matrix: Sparse TF-IDF matrix.
            embedding_matrix: Dense embedding matrix (optional).

        Returns:
            Dense combined feature matrix.
        """
        if embedding_matrix is None:
            if issparse(tfidf_matrix):
                return tfidf_matrix.toarray()
            return np.array(tfidf_matrix)

        tfidf_dense = tfidf_matrix.toarray() if issparse(tfidf_matrix) else np.array(tfidf_matrix)
        combined = np.hstack([tfidf_dense, embedding_matrix])
        logger.info("Combined features: %d dims", combined.shape[1])
        return combined
