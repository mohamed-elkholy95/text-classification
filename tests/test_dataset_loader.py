"""Tests for dataset_loader module."""
import pytest
import numpy as np
import pandas as pd
from src.data.dataset_loader import generate_synthetic_data, load_dataset, get_dataset_stats


class TestGenerateSyntheticData:
    def test_returns_dataframe(self):
        df = generate_synthetic_data(n_samples=100)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = generate_synthetic_data(n_samples=100)
        assert "text" in df.columns
        assert "label" in df.columns

    def test_correct_shape(self):
        df = generate_synthetic_data(n_samples=150)
        assert df.shape[0] == 150

    def test_reproducible(self):
        df1 = generate_synthetic_data(n_samples=100, seed=42)
        df2 = generate_synthetic_data(n_samples=100, seed=42)
        assert df1["text"].tolist() == df2["text"].tolist()

    def test_binary_classes(self):
        df = generate_synthetic_data(n_samples=100, n_classes=2)
        assert set(df["label"].unique()).issubset({0, 1})


class TestLoadDataset:
    def test_load_synthetic(self):
        df = load_dataset("synthetic", n_samples=100)
        assert len(df) == 100

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent")


class TestGetDatasetStats:
    def test_stats_keys(self):
        df = generate_synthetic_data(n_samples=100)
        stats = get_dataset_stats(df)
        assert "n_samples" in stats
        assert "n_classes" in stats
        assert "class_distribution" in stats
        assert "avg_text_length" in stats

    def test_stats_values(self):
        df = generate_synthetic_data(n_samples=100)
        stats = get_dataset_stats(df)
        assert stats["n_samples"] == 100
        assert stats["n_classes"] >= 1
