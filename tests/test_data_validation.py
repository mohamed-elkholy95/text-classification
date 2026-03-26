"""Tests for dataset validation and preprocessor edge cases.

These tests verify that the pipeline handles messy real-world data
gracefully — NaN values, empty strings, missing columns, and class
imbalance are all common in production datasets.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.dataset_loader import validate_dataset
from src.data.preprocessor import TextPreprocessor


# ─── Preprocessor edge-case tests ────────────────────────────────────────────


class TestPreprocessorEdgeCases:
    """Verify clean_text handles non-standard inputs without crashing."""

    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()

    def test_clean_text_with_none(self, preprocessor):
        """None values should produce an empty string, not raise."""
        assert preprocessor.clean_text(None) == ""

    def test_clean_text_with_nan(self, preprocessor):
        """Float NaN (common in pandas) should produce an empty string."""
        assert preprocessor.clean_text(float("nan")) == ""

    def test_clean_text_with_number(self, preprocessor):
        """Numeric inputs should be coerced to string."""
        result = preprocessor.clean_text(42)
        assert isinstance(result, str)
        assert "42" in result

    def test_clean_text_with_empty_string(self, preprocessor):
        """Empty string is a valid input — should return empty."""
        assert preprocessor.clean_text("") == ""

    def test_clean_dataframe_with_nan_values(self, preprocessor):
        """DataFrames with NaN text values should not raise."""
        df = pd.DataFrame({
            "text": ["hello world", np.nan, "good product", None],
            "label": [0, 1, 0, 1],
        })
        result = preprocessor.clean_dataframe(df)
        # NaN rows should become empty strings, not raise
        assert result["text"].isna().sum() == 0
        assert result.loc[1, "text"] == ""

    def test_clean_dataframe_missing_column_raises(self, preprocessor):
        """Requesting a non-existent column should raise KeyError."""
        df = pd.DataFrame({"content": ["hello"], "label": [0]})
        with pytest.raises(KeyError, match="text"):
            preprocessor.clean_dataframe(df, text_col="text")


# ─── Dataset validation tests ────────────────────────────────────────────────


class TestDatasetValidation:
    """Verify validate_dataset catches common data quality issues."""

    def test_valid_dataset_passes(self):
        """A clean dataset should pass with no errors or warnings."""
        df = pd.DataFrame({
            "text": [f"sample text {i}" for i in range(20)],
            "label": [0] * 10 + [1] * 10,
        })
        report = validate_dataset(df)
        assert report["valid"] is True
        assert len(report["errors"]) == 0

    def test_missing_text_column_is_error(self):
        """Missing text column is a fatal error."""
        df = pd.DataFrame({"content": ["hello"], "label": [0]})
        report = validate_dataset(df)
        assert report["valid"] is False
        assert any("text" in e for e in report["errors"])

    def test_missing_label_column_is_error(self):
        """Missing label column is a fatal error."""
        df = pd.DataFrame({"text": ["hello"], "target": [0]})
        report = validate_dataset(df)
        assert report["valid"] is False
        assert any("label" in e for e in report["errors"])

    def test_null_text_values_are_errors(self):
        """Null values in the text column should be flagged as errors."""
        df = pd.DataFrame({
            "text": ["hello", None, "world"],
            "label": [0, 1, 0],
        })
        report = validate_dataset(df)
        assert report["valid"] is False
        assert report["stats"]["null_text"] == 1

    def test_null_label_values_are_errors(self):
        """Null values in the label column should be flagged as errors."""
        df = pd.DataFrame({
            "text": ["hello", "world", "test"],
            "label": [0, np.nan, 1],
        })
        report = validate_dataset(df)
        assert report["valid"] is False
        assert report["stats"]["null_label"] == 1

    def test_empty_text_values_are_warnings(self):
        """Empty strings are non-fatal but worth flagging."""
        df = pd.DataFrame({
            "text": ["hello", "", "   ", "world", "test", "ok"],
            "label": [0, 0, 0, 1, 1, 1],
        })
        report = validate_dataset(df)
        assert report["valid"] is True  # warnings are non-fatal
        assert report["stats"]["empty_text"] >= 2

    def test_duplicates_are_warnings(self):
        """Duplicate rows should produce a warning."""
        df = pd.DataFrame({
            "text": ["hello", "hello", "world", "world", "test", "ok"],
            "label": [0, 0, 1, 1, 0, 1],
        })
        report = validate_dataset(df)
        assert report["stats"]["duplicates"] == 2

    def test_class_imbalance_warning(self):
        """Classes with very few samples should trigger a warning."""
        df = pd.DataFrame({
            "text": [f"text {i}" for i in range(12)],
            "label": [0] * 10 + [1] * 2,
        })
        report = validate_dataset(df, min_samples_per_class=5)
        assert any("Class" in w and "1" in w for w in report["warnings"])

    def test_custom_column_names(self):
        """Validation should respect custom column name arguments."""
        df = pd.DataFrame({
            "content": ["hello", "world"],
            "target": [0, 1],
        })
        report = validate_dataset(df, text_col="content", label_col="target")
        assert report["valid"] is True
