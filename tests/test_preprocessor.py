"""Tests for preprocessor module."""
import pytest
import numpy as np
import pandas as pd
from src.data.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    def test_clean_text_lowercase(self):
        prep = TextPreprocessor()
        assert prep.clean_text("HELLO WORLD") == "hello world"

    def test_clean_text_removes_urls(self):
        prep = TextPreprocessor()
        result = prep.clean_text("Visit https://example.com now")
        assert "https" not in result

    def test_clean_text_removes_mentions(self):
        prep = TextPreprocessor()
        result = prep.clean_text("@user hello world")
        assert "@user" not in result

    def test_clean_text_removes_special_chars(self):
        prep = TextPreprocessor()
        result = prep.clean_text("Hello!!! $$$%^&")
        assert "$$$" not in result

    def test_tokenize(self):
        prep = TextPreprocessor()
        assert prep.tokenize("hello world") == ["hello", "world"]

    def test_clean_dataframe(self):
        prep = TextPreprocessor()
        df = pd.DataFrame({"text": ["HELLO WORLD", "Test 123"]})
        cleaned = prep.clean_dataframe(df)
        assert cleaned["text"].iloc[0] == "hello world"

    def test_encode_decode_labels(self):
        prep = TextPreprocessor()
        labels = np.array(["spam", "ham", "spam", "ham"])
        encoded = prep.encode_labels(labels)
        decoded = prep.decode_labels(encoded)
        assert list(decoded) == list(labels)

    def test_fit_transform(self):
        prep = TextPreprocessor()
        df = pd.DataFrame({"text": ["HELLO WORLD", "GOOD DAY"], "label": ["pos", "neg"]})
        cleaned, labels = prep.fit_transform(df)
        assert len(cleaned) == 2
        assert len(labels) == 2
        assert len(prep.classes) == 2

    def test_classes_property_unfitted(self):
        prep = TextPreprocessor()
        assert len(prep.classes) == 0
