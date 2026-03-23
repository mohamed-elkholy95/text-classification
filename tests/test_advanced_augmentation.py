"""Tests for advanced augmentation functions: random_insertion and augment_dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.data.augmentor import random_insertion, augment_dataset, synonym_replacement


class TestRandomInsertion:
    def test_short_text_unchanged(self):
        result = random_insertion("hi", n=1, seed=42)
        assert result == "hi"

    def test_single_word_unchanged(self):
        result = random_insertion("hello", n=1, seed=42)
        assert result == "hello"

    def test_insertion_increases_length(self):
        text = "the cat sat on the mat"
        result = random_insertion(text, n=3, seed=42)
        # After insertions, the result should have more words
        result_words = result.split()
        original_words = text.split()
        assert len(result_words) > len(original_words)

    def test_zero_insertions_returns_original(self):
        text = "the cat sat on the mat"
        result = random_insertion(text, n=0, seed=42)
        assert result == text

    def test_reproducibility(self):
        text = "the cat sat on the mat"
        r1 = random_insertion(text, n=2, seed=99)
        r2 = random_insertion(text, n=2, seed=99)
        assert r1 == r2

    def test_different_seeds_produce_different_results(self):
        text = "the cat sat on the mat"
        r1 = random_insertion(text, n=3, seed=1)
        r2 = random_insertion(text, n=3, seed=2)
        # With high probability these differ
        assert r1 != r2


class TestAugmentDataset:
    def test_balances_imbalanced_dataset(self):
        df = pd.DataFrame({
            "text": ["good movie", "bad film", "terrible plot"],
            "label": ["pos", "neg", "neg"],
        })
        balanced = augment_dataset(df, augment_fn=synonym_replacement)
        # neg had 2, pos had 1 -> should add 1 pos sample -> 3 total
        assert balanced["label"].value_counts()["pos"] == 2
        assert balanced["label"].value_counts()["neg"] == 2

    def test_balanced_dataset_unchanged(self):
        df = pd.DataFrame({
            "text": ["good", "bad"],
            "label": ["pos", "neg"],
        })
        balanced = augment_dataset(df, augment_fn=synonym_replacement)
        assert len(balanced) == 2

    def test_missing_text_column_raises(self):
        df = pd.DataFrame({"label": ["pos", "neg"]})
        with pytest.raises(ValueError, match="Column 'text' not found"):
            augment_dataset(df)

    def test_missing_label_column_raises(self):
        df = pd.DataFrame({"text": ["good", "bad"]})
        with pytest.raises(ValueError, match="Column 'label' not found"):
            augment_dataset(df)

    def test_original_dataframe_not_modified(self):
        df = pd.DataFrame({
            "text": ["good movie", "bad film", "terrible plot"],
            "label": ["pos", "neg", "neg"],
        })
        original_len = len(df)
        _ = augment_dataset(df, augment_fn=synonym_replacement)
        assert len(df) == original_len

    def test_large_imbalance_balanced(self):
        def simple_augment(text, seed=42):
            # Simple augmentation that always changes text
            return text + " augmented"

        df = pd.DataFrame({
            "text": ["good movie"] + ["bad film"] * 9,
            "label": ["pos"] + ["neg"] * 9,
        })
        balanced = augment_dataset(df, augment_fn=simple_augment)
        # pos had 1, neg had 10 -> should add 9 pos samples
        assert balanced["label"].value_counts()["pos"] == 10
        assert balanced["label"].value_counts()["neg"] == 10
