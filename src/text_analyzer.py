"""Text feature extraction and linguistic analysis.

Why Text Statistics Matter for Classification:
─────────────────────────────────────────────────
Before feeding text into a classifier, it's valuable to understand the
*characteristics* of your corpus and whether those characteristics
differ across classes.  A model might be implicitly relying on document
length, vocabulary richness, or readability — features that are easily
computed but often overlooked.

For example:
- **Spam vs ham** emails tend to differ in average word count and
  vocabulary richness.
- **Sentiment analysis** can be confounded by document length (longer
  reviews tend to be more polarised).
- **Author attribution** often exploits sentence-length distributions
  and hapax legomena (words that appear only once).

This module provides reusable, class-conditional text statistics that
feed into both EDA (exploratory data analysis) and feature engineering.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TextAnalyzer:
    """Compute linguistic statistics for text datasets.

    Provides document-level metrics (average word count, sentence
    length, vocabulary richness, readability) and supports computing
    these statistics *conditionally* per class label.

    Attributes:
        texts: Stored list of preprocessed text documents.

    Example::

        >>> from src.text_analyzer import TextAnalyzer
        >>> analyzer = TextAnalyzer(["Hello world!", "This is a test.", "Another sentence here."])
        >>> stats = analyzer.compute_statistics()
        >>> round(stats["avg_word_count"], 2)
        3.33
        >>> cond = analyzer.class_conditional_stats(
        ...     labels=[0, 1, 1],
        ...     class_names=["spam", "ham"],
        ... )
        >>> cond["ham"]["avg_word_count"]
        3.5
    """

    def __init__(self, texts: Sequence[str]) -> None:
        """Initialise the analyser with a corpus.

        Args:
            texts: Iterable of string documents.  Empty strings are
                allowed; they simply contribute zero-count statistics.

        Raises:
            ValueError: If *texts* is empty.
        """
        if not texts:
            raise ValueError("texts must contain at least one document.")
        self.texts: List[str] = list(texts)

    # ------------------------------------------------------------------
    # Single-document helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize_words(text: str) -> List[str]:
        """Lowercase and split on non-alphanumeric characters.

        A simple regex tokeniser is sufficient for many classification
        tasks and avoids heavy dependencies (spaCy, NLTK).
        """
        return re.findall(r"[a-zA-Z0-9']+", text.lower())

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences on ``.``, ``!``, or ``?``."""
        # Filter out empty strings that arise from trailing punctuation
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    # ------------------------------------------------------------------
    # Document-level metrics
    # ------------------------------------------------------------------

    @staticmethod
    def avg_word_count(text: str) -> float:
        """Compute the average word count of a single document.

        For a single document the "average" is just the count, but the
        method name mirrors the per-corpus API for consistency.

        Args:
            text: A string document.

        Returns:
            Number of tokens in the document.
        """
        return float(len(TextAnalyzer._tokenize_words(text)))

    @staticmethod
    def avg_sentence_length(text: str) -> float:
        """Compute the average number of words per sentence.

        Args:
            text: A string document.

        Returns:
            Mean words per sentence.  Returns 0.0 for empty text.
        """
        sentences = TextAnalyzer._split_sentences(text)
        if not sentences:
            return 0.0
        total_words = sum(len(TextAnalyzer._tokenize_words(s)) for s in sentences)
        return total_words / len(sentences)

    @staticmethod
    def vocabulary_richness(text: str) -> float:
        """Type-token ratio (unique words / total words).

        A higher ratio indicates a richer, more diverse vocabulary.
        Note that TTR is length-dependent (shorter texts tend to have
        higher TTR), so compare documents of similar length.

        Args:
            text: A string document.

        Returns:
            Type-token ratio in [0, 1].  Returns 0.0 for empty text.
        """
        tokens = TextAnalyzer._tokenize_words(text)
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    @staticmethod
    def readability(text: str) -> float:
        """Simplified Flesch-Kincaid readability score.

        The **Flesch Reading Ease** formula estimates how easy a text is
        to read on a 0–100 scale (higher = easier).  The classic formula
        requires syllable counts, which are expensive to compute
        accurately without a dictionary.  This simplified version uses
        the proxy that average word length (in characters) correlates
        strongly with syllable count.

        Formula (simplified):
            ``206.835 − 1.015 × (total_words / n_sentences)
                      − 60.0 × (total_chars / total_words)``

        Args:
            text: A string document.

        Returns:
            Flesch Reading Ease score.  Returns 0.0 for empty text.
        """
        sentences = TextAnalyzer._split_sentences(text)
        tokens = TextAnalyzer._tokenize_words(text)
        if not tokens or not sentences:
            return 0.0

        n_sentences = len(sentences)
        total_words = len(tokens)
        # Use character count of original alphabetic tokens as a
        # syllable-count proxy — fast and reasonably correlated.
        total_chars = sum(len(t) for t in tokens)
        avg_sentence = total_words / n_sentences
        avg_word_len = total_chars / total_words

        score = 206.835 - 1.015 * avg_sentence - 60.0 * avg_word_len
        # Clamp to [0, 100]
        return max(0.0, min(100.0, score))

    @staticmethod
    def word_frequency(text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """Return the most common words in a document.

        Args:
            text: A string document.
            top_n: Number of top words to return.  Defaults to 10.

        Returns:
            List of ``(word, count)`` tuples sorted by descending count.
        """
        tokens = TextAnalyzer._tokenize_words(text)
        return Counter(tokens).most_common(top_n)

    # ------------------------------------------------------------------
    # Corpus-level aggregations
    # ------------------------------------------------------------------

    def compute_statistics(
        self,
        top_n_words: int = 20,
    ) -> Dict[str, Any]:
        """Compute summary statistics across the full corpus.

        For each metric the mean and standard deviation across documents
        are reported, along with the aggregate word-frequency table.

        Args:
            top_n_words: Number of most-common words in the global
                frequency table.  Defaults to 20.

        Returns:
            Dictionary containing:
            - ``"avg_word_count"``: mean words per document.
            - ``"std_word_count"``: std of word counts.
            - ``"avg_sentence_length"``: mean of per-doc averages.
            - ``"std_sentence_length"``: std of sentence lengths.
            - ``"avg_vocabulary_richness"``: mean type-token ratio.
            - ``"std_vocabulary_richness"``: std of TTR.
            - ``"avg_readability"``: mean Flesch score.
            - ``"std_readability"``: std of Flesch score.
            - ``"n_documents"``: total number of documents.
            - ``"top_words"``: list of (word, count) tuples.
        """
        word_counts = []
        sent_lengths = []
        ttrs = []
        flesch_scores = []
        global_counter: Counter = Counter()

        for text in self.texts:
            tokens = self._tokenize_words(text)
            word_counts.append(len(tokens))
            sent_lengths.append(self.avg_sentence_length(text))
            ttrs.append(self.vocabulary_richness(text))
            flesch_scores.append(self.readability(text))
            global_counter.update(tokens)

        wc = np.array(word_counts, dtype=float)
        sl = np.array(sent_lengths, dtype=float)
        vr = np.array(ttrs, dtype=float)
        fs = np.array(flesch_scores, dtype=float)

        report: Dict[str, Any] = {
            "n_documents": len(self.texts),
            "avg_word_count": round(float(wc.mean()), 2),
            "std_word_count": round(float(wc.std()), 2),
            "avg_sentence_length": round(float(sl.mean()), 2),
            "std_sentence_length": round(float(sl.std()), 2),
            "avg_vocabulary_richness": round(float(vr.mean()), 4),
            "std_vocabulary_richness": round(float(vr.std()), 4),
            "avg_readability": round(float(fs.mean()), 2),
            "std_readability": round(float(fs.std()), 2),
            "top_words": global_counter.most_common(top_n_words),
        }

        logger.info(
            "Corpus stats: %d docs, avg %.1f words/doc, readability %.1f",
            report["n_documents"],
            report["avg_word_count"],
            report["avg_readability"],
        )
        return report

    def class_conditional_stats(
        self,
        labels: Sequence[int],
        class_names: Optional[Dict[int, str]] = None,
        top_n_words: int = 20,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute statistics separately for each class label.

        This is useful for identifying linguistic differences between
        classes that a classifier might be exploiting (or missing).

        Args:
            labels: Integer class labels, one per document.
            class_names: Optional mapping from label int to readable
                name.  Defaults to ``str(label)``.
            top_n_words: Number of top words per class.  Defaults to 20.

        Returns:
            Nested dict ``{class_name: {statistic: value}}`` mirroring
            the structure of :meth:`compute_statistics`.

        Raises:
            ValueError: If *labels* length does not match the number of
                documents.
        """
        labels = list(labels)
        if len(labels) != len(self.texts):
            raise ValueError(
                f"len(labels)={len(labels)} != len(texts)={len(self.texts)}."
            )

        class_names = class_names or {}
        grouped: Dict[int, List[str]] = {}
        for text, label in zip(self.texts, labels):
            grouped.setdefault(label, []).append(text)

        result: Dict[str, Dict[str, Any]] = {}
        for label, texts in grouped.items():
            name = class_names.get(label, str(label))
            analyzer = TextAnalyzer(texts)
            result[name] = analyzer.compute_statistics(top_n_words=top_n_words)
            result[name]["n_documents"] = len(texts)
            logger.info("Class '%s': %d documents.", name, len(texts))

        return result
