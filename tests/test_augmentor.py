"""Tests for augmentation."""
import pytest
from src.data.augmentor import synonym_replacement, random_deletion, random_swap


class TestSynonymReplacement:
    def test_returns_string(self):
        result = synonym_replacement("great product", seed=42)
        assert isinstance(result, str)

    def test_same_seed_same_result(self):
        r1 = synonym_replacement("great product", seed=42)
        r2 = synonym_replacement("great product", seed=42)
        assert r1 == r2

    def test_short_text_unchanged(self):
        result = synonym_replacement("hi", seed=42)
        assert "hi" in result


class TestRandomDeletion:
    def test_returns_string(self):
        result = random_deletion("hello world test", p=0.1, seed=42)
        assert isinstance(result, str)

    def test_zero_p_returns_same(self):
        text = "hello world test"
        result = random_deletion(text, p=0.0, seed=42)
        assert result == text

    def test_short_text_unchanged(self):
        result = random_deletion("hi", p=0.5, seed=42)
        assert "hi" in result


class TestRandomSwap:
    def test_returns_string(self):
        result = random_swap("hello world", seed=42)
        assert isinstance(result, str)

    def test_single_word_unchanged(self):
        result = random_swap("hello", seed=42)
        assert result == "hello"

    def test_same_seed_same_result(self):
        r1 = random_swap("hello world", seed=42)
        r2 = random_swap("hello world", seed=42)
        assert r1 == r2
