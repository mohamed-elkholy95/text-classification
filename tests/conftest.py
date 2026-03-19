"""Shared test fixtures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data.dataset_loader import generate_synthetic_data, get_dataset_stats
from src.data.preprocessor import TextPreprocessor
from src.features.tfidf_features import TfidfFeatureExtractor


@pytest.fixture
def sample_df():
    return generate_synthetic_data(n_samples=200, n_classes=2, seed=42)


@pytest.fixture
def preprocessor():
    return TextPreprocessor(seed=42)


@pytest.fixture
def cleaned_data(sample_df, preprocessor):
    return preprocessor.clean_dataframe(sample_df)


@pytest.fixture
def tfidf_extractor():
    return TfidfFeatureExtractor(max_features=500, min_df=1)


@pytest.fixture
def tfidf_matrix(cleaned_data, tfidf_extractor):
    return tfidf_extractor.fit_transform(cleaned_data["text"].tolist())


@pytest.fixture
def sample_texts():
    return ["This is great", "Terrible product", "Amazing work", "Worst experience ever",
            "Good quality", "Bad service", "Love it so much", "Hate this thing"]


@pytest.fixture
def sample_labels():
    return np.array([1, 0, 1, 0, 1, 0, 1, 0])


@pytest.fixture
def sample_labels_long():
    return np.random.default_rng(42).integers(0, 2, 200)
