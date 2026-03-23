"""Tests for TextAnalyzer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.text_analyzer import TextAnalyzer


class TestTextAnalyzerInit:
    def test_init_with_list(self):
        analyzer = TextAnalyzer(["Hello world", "Test text"])
        assert len(analyzer.texts) == 2

    def test_init_empty_raises(self):
        with pytest.raises(ValueError, match="texts must contain at least one"):
            TextAnalyzer([])


class TestTextAnalyzerStaticMethods:
    def test_avg_word_count(self):
        assert TextAnalyzer.avg_word_count("Hello world foo") == 3.0
        assert TextAnalyzer.avg_word_count("") == 0.0

    def test_avg_sentence_length(self):
        result = TextAnalyzer.avg_sentence_length("Hello world. This is a test.")
        # "Hello world" = 2 words, "This is a test" = 4 words; avg = 3.0
        assert result == 3.0

    def test_avg_sentence_length_empty(self):
        assert TextAnalyzer.avg_sentence_length("") == 0.0

    def test_vocabulary_richness(self):
        # All unique words => ratio 1.0
        assert TextAnalyzer.vocabulary_richness("cat dog bird") == 1.0
        # Half unique => ratio 0.5
        assert TextAnalyzer.vocabulary_richness("cat cat dog dog") == 0.5

    def test_vocabulary_richness_empty(self):
        assert TextAnalyzer.vocabulary_richness("") == 0.0

    def test_readability_returns_float(self):
        result = TextAnalyzer.readability("This is a simple sentence for testing.")
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0

    def test_readability_empty(self):
        assert TextAnalyzer.readability("") == 0.0

    def test_word_frequency(self):
        freq = TextAnalyzer.word_frequency("cat dog cat bird cat")
        assert freq[0] == ("cat", 3)
        assert len(freq) <= 10


class TestTextAnalyzerComputeStatistics:
    def test_compute_statistics_keys(self):
        analyzer = TextAnalyzer(["Hello world.", "Another test sentence here."])
        stats = analyzer.compute_statistics()

        expected_keys = [
            "n_documents", "avg_word_count", "std_word_count",
            "avg_sentence_length", "std_sentence_length",
            "avg_vocabulary_richness", "std_vocabulary_richness",
            "avg_readability", "std_readability", "top_words",
        ]
        for key in expected_keys:
            assert key in stats

    def test_compute_statistics_n_documents(self):
        analyzer = TextAnalyzer(["one", "two", "three"])
        stats = analyzer.compute_statistics()
        assert stats["n_documents"] == 3

    def test_compute_statistics_top_words(self):
        analyzer = TextAnalyzer(["cat dog cat", "dog bird"])
        stats = analyzer.compute_statistics()
        top_words = stats["top_words"]
        assert top_words[0][0] == "cat"  # cat appears 2 times
        assert top_words[0][1] == 2


class TestTextAnalyzerClassConditional:
    def test_class_conditional_stats(self):
        analyzer = TextAnalyzer(["good movie", "bad film", "terrible plot"])
        result = analyzer.class_conditional_stats(
            labels=[0, 1, 1],
            class_names={0: "pos", 1: "neg"},
        )
        assert "pos" in result
        assert "neg" in result
        assert result["pos"]["n_documents"] == 1
        assert result["neg"]["n_documents"] == 2

    def test_class_conditional_stats_mismatch_raises(self):
        analyzer = TextAnalyzer(["hello", "world"])
        with pytest.raises(ValueError, match="len\\(labels\\).*!=.*len\\(texts\\)"):
            analyzer.class_conditional_stats(labels=[0])
