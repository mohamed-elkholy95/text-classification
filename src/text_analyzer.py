"""Text analysis utilities for exploratory data analysis."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TextAnalyzer:
    """Exploratory text analysis for classification datasets.

    Computes word counts, character counts, vocabulary statistics,
    and per-class summaries for a corpus of texts.

    Example::

        >>> analyzer = TextAnalyzer()
        >>> stats = analyzer.analyze("Hello world this is a test")
        >>> stats["word_count"]
        6
    """

    def analyze(self, text: str) -> Dict[str, Any]:
        """Compute statistics for a single text.

        Args:
            text: Input string.

        Returns:
            Dictionary with ``word_count``, ``char_count``,
            ``unique_words``, ``avg_word_length``, ``words``.
        """
        if not text or not text.strip():
            return {
                "word_count": 0,
                "char_count": 0,
                "unique_words": 0,
                "avg_word_length": 0.0,
                "words": [],
            }
        words = text.split()
        word_lengths = [len(w) for w in words]
        return {
            "word_count": len(words),
            "char_count": len(text),
            "unique_words": len(set(w.lower() for w in words)),
            "avg_word_length": float(np.mean(word_lengths)) if word_lengths else 0.0,
            "words": words,
        }

    def analyze_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Compute statistics for multiple texts.

        Args:
            texts: List of input strings.

        Returns:
            List of per-text statistic dictionaries.
        """
        return [self.analyze(t) for t in texts]

    def class_conditional_stats(
        self,
        texts: List[str],
        labels: np.ndarray,
    ) -> Dict[int, Dict[str, Any]]:
        """Compute aggregated stats per class.

        Args:
            texts: List of input strings.
            labels: Array of integer class labels.

        Returns:
            Dictionary keyed by class label, each containing
            ``count``, ``avg_word_count``, ``avg_char_count``,
            ``most_common_words``.
        """
        labels_arr = np.asarray(labels)
        unique_labels = sorted(set(labels_arr.tolist()))
        grouped: Dict[int, List[str]] = {lbl: [] for lbl in unique_labels}
        for txt, lbl in zip(texts, labels_arr):
            grouped[int(lbl)].append(txt)

        result: Dict[int, Dict[str, Any]] = {}
        for lbl, group_texts in grouped.items():
            stats_list = self.analyze_texts(group_texts)
            wc = [s["word_count"] for s in stats_list]
            cc = [s["char_count"] for s in stats_list]
            all_words = []
            for s in stats_list:
                all_words.extend(w.lower() for w in s["words"])
            result[lbl] = {
                "count": len(group_texts),
                "avg_word_count": float(np.mean(wc)) if wc else 0.0,
                "avg_char_count": float(np.mean(cc)) if cc else 0.0,
                "most_common_words": Counter(all_words).most_common(5),
            }
        return result
