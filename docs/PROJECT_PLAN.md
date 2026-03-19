# Project Plan — Text Classification Pipeline

> **Duration:** 10 days  
> **Goal:** Build a production-ready, end-to-end text classification system covering data collection, feature engineering, model training, hyperparameter tuning, REST API deployment, and a Streamlit dashboard.

---

## Table of Contents

1. [Phase 1: Data Collection (Days 1–2)](#phase-1-data-collection)
2. [Phase 2: Feature Engineering (Days 2–3)](#phase-2-feature-engineering)
3. [Phase 3: Model Training (Days 3–5)](#phase-3-model-training)
4. [Phase 4: Hyperparameter Tuning (Days 5–6)](#phase-4-hyperparameter-tuning)
5. [Phase 5: API Deployment (Days 6–8)](#phase-5-api-deployment)
6. [Phase 6: Dashboard (Days 8–9)](#phase-6-dashboard)
7. [Phase 7: Evaluation (Days 9–10)](#phase-7-evaluation)
8. [File Structure](#file-structure)
9. [Dependencies](#dependencies)
10. [Success Criteria](#success-criteria)

---

## Phase 1: Data Collection (Days 1–2)

### Objective
Ingest multiple public datasets for sentiment analysis and spam detection, apply data augmentation to increase training diversity, and build a robust preprocessing pipeline.

### Datasets

| Dataset | Task | Size | Source |
|---|---|---|---|
| IMDB Movie Reviews | Sentiment (binary) | 50k | HuggingFace `datasets` |
| SMS Spam Collection | Spam detection | 5.5k | UCI ML Repository |
| Twitter Sentiment140 | Sentiment (binary) | 1.6M | Kaggle |
| Custom domain dataset | User-defined | variable | Local CSV |

### Files

#### `src/data/dataset_loader.py`

```python
from typing import Literal, Optional
import pandas as pd
from datasets import load_dataset as hf_load_dataset

DatasetSplit = Literal["train", "validation", "test"]
SupportedDataset = Literal["imdb", "sms_spam", "twitter_sentiment", "custom"]

def load_dataset(
    name: SupportedDataset,
    split: DatasetSplit,
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a named dataset and return as a standardized DataFrame.

    Returns:
        DataFrame with columns: ['text', 'label', 'label_name']
        where label is an integer and label_name is human-readable.

    Raises:
        ValueError: If dataset name is not supported.
        FileNotFoundError: If 'custom' dataset and data_dir is missing.
    """
    ...

def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Compute class distribution, average text length, vocabulary size.

    Returns:
        {
          "n_samples": int,
          "class_distribution": dict[str, int],
          "avg_tokens": float,
          "max_tokens": int,
          "vocab_size": int,
        }
    """
    ...

def merge_datasets(
    datasets: list[pd.DataFrame],
    stratify: bool = True,
) -> pd.DataFrame:
    """Merge multiple DataFrames, optionally stratified by label."""
    ...
```

#### `src/data/augmentor.py`

```python
from typing import Literal, Optional
import random

AugmentMethod = Literal[
    "synonym_replacement",
    "back_translation",
    "random_deletion",
    "random_swap",
    "contextual_insertion",
]

def augment_text(
    text: str,
    method: AugmentMethod,
    n_augments: int = 1,
    p: float = 0.1,
    lang_pair: str = "en-de",
    seed: Optional[int] = None,
) -> list[str]:
    """
    Augment a single text sample using the specified strategy.

    Args:
        text:       Original text to augment.
        method:     Augmentation method (see AugmentMethod).
        n_augments: Number of augmented samples to generate.
        p:          Probability for word-level operations (deletion/swap).
        lang_pair:  Language pair for back-translation (e.g., "en-de").
        seed:       Random seed for reproducibility.

    Returns:
        List of n_augments augmented strings.

    Examples:
        >>> augment_text("The food was great", "synonym_replacement", n_augments=2)
        ["The nutriment was great", "The food was tremendous"]
    """
    ...

def augment_dataset(
    df: pd.DataFrame,
    methods: list[AugmentMethod],
    n_augments_per_method: int = 1,
    label_col: str = "label",
    text_col: str = "text",
    minority_only: bool = True,
) -> pd.DataFrame:
    """
    Augment an entire dataset. If minority_only=True, only augments
    the minority class(es) to address class imbalance.

    Returns:
        Augmented DataFrame appended to original.
    """
    ...
```

#### `src/data/preprocessor.py`

```python
import re
from typing import Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class TextPreprocessor:
    """
    Configurable text preprocessing pipeline. Each step is toggleable
    to allow experiments with different preprocessing levels.

    Usage:
        preprocessor = TextPreprocessor(
            lowercase=True,
            remove_urls=True,
            remove_punctuation=True,
            remove_stopwords=True,
            lemmatize=True,
        )
        cleaned = preprocessor.process("Check out https://example.com!")
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        stemming: bool = False,
        lemmatize: bool = True,
        min_token_length: int = 2,
        custom_stopwords: Optional[list[str]] = None,
        language: str = "english",
    ) -> None: ...

    def clean(self, text: str) -> str:
        """
        Apply HTML removal, URL removal, special character cleaning.
        Does NOT remove stopwords or apply stemming/lemmatization.
        """
        ...

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize cleaned text using NLTK word_tokenize.
        Handles contractions (don't → do, n't) and punctuation.
        """
        ...

    def normalize(self, tokens: list[str]) -> list[str]:
        """
        Apply stemming or lemmatization depending on config.
        Stemming is faster; lemmatization produces real words.
        """
        ...

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove NLTK + custom stopwords from token list."""
        ...

    def process(self, text: str) -> str:
        """Apply full pipeline and return cleaned string."""
        ...

    def process_batch(
        self,
        texts: list[str],
        n_jobs: int = -1,
    ) -> list[str]:
        """Parallel processing using joblib for large datasets."""
        ...
```

### Deliverables
- [ ] `load_dataset()` works for all 4 named datasets
- [ ] `augment_text()` implements all 5 augmentation strategies
- [ ] `TextPreprocessor.process()` handles edge cases (empty, emoji, HTML)
- [ ] Dataset stats logged to `logs/data_stats.json`

---

## Phase 2: Feature Engineering (Days 2–3)

### Objective
Extract numerical features from text using classical (TF-IDF) and neural (Word2Vec, GloVe, BERT) representations, then combine them for downstream models.

### Files

#### `src/features/tfidf_features.py`

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix
import numpy as np
import joblib

def build_tfidf(
    corpus: list[str],
    max_features: int = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
    save_path: Optional[str] = None,
) -> tuple[spmatrix, TfidfVectorizer]:
    """
    Fit TF-IDF vectorizer on corpus and transform it.

    Args:
        corpus:       List of preprocessed strings.
        max_features: Maximum vocabulary size.
        ngram_range:  Unigrams + bigrams by default.
        min_df:       Minimum document frequency.
        max_df:       Maximum document frequency (removes stop-ish words).
        sublinear_tf: Apply log(1 + tf) scaling.
        save_path:    If provided, persist vectorizer with joblib.

    Returns:
        (feature_matrix, fitted_vectorizer)
    """
    ...

def transform_tfidf(
    texts: list[str],
    vectorizer: TfidfVectorizer,
) -> spmatrix:
    """Transform new texts using a pre-fitted vectorizer."""
    ...

def load_tfidf(path: str) -> TfidfVectorizer:
    """Load a persisted TF-IDF vectorizer from disk."""
    ...

def get_top_features(
    vectorizer: TfidfVectorizer,
    n: int = 20,
) -> list[tuple[str, float]]:
    """Return the top-n most discriminating features by IDF weight."""
    ...
```

#### `src/features/embedding_features.py`

```python
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Literal

EmbeddingModel = Literal["bert-base-uncased", "distilbert-base-uncased", "roberta-base"]

def get_word_embeddings(
    texts: list[str],
    model: EmbeddingModel = "bert-base-uncased",
    pooling: Literal["cls", "mean", "max"] = "mean",
    batch_size: int = 32,
    device: str = "auto",
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract sentence embeddings from a transformer model.

    Args:
        texts:      List of input strings.
        model:      HuggingFace model name.
        pooling:    How to pool token embeddings into sentence embedding.
                    'cls' = [CLS] token, 'mean' = average, 'max' = max-pool.
        batch_size: Inference batch size.
        device:     'auto' selects CUDA if available, else CPU.
        normalize:  L2-normalize embeddings (recommended for cosine similarity).

    Returns:
        ndarray of shape (len(texts), hidden_dim).
    """
    ...

def get_glove_embeddings(
    texts: list[str],
    glove_path: str,
    dim: int = 300,
    pooling: Literal["mean", "max"] = "mean",
) -> np.ndarray:
    """
    Average GloVe word vectors across tokens in each text.

    Args:
        glove_path: Path to GloVe .txt file (e.g., glove.6B.300d.txt).
        dim:        Embedding dimension (50, 100, 200, 300).

    Returns:
        ndarray of shape (len(texts), dim).
    """
    ...

def load_glove(path: str, dim: int = 300) -> dict[str, np.ndarray]:
    """Load GloVe vectors from text file into a word→vector dict."""
    ...
```

#### `src/features/feature_combiner.py`

```python
import numpy as np
from scipy.sparse import hstack, spmatrix
from sklearn.preprocessing import StandardScaler

def combine_features(
    tfidf: spmatrix,
    embeddings: np.ndarray,
    scale_embeddings: bool = True,
) -> np.ndarray:
    """
    Horizontally stack TF-IDF (sparse) with dense embedding features.

    The sparse TF-IDF matrix is converted to dense; embeddings are
    optionally standardized to zero mean and unit variance to prevent
    TF-IDF features from dominating due to scale differences.

    Args:
        tfidf:            Sparse matrix (n_samples, tfidf_features).
        embeddings:       Dense array (n_samples, embed_dim).
        scale_embeddings: Apply StandardScaler to embedding columns.

    Returns:
        Dense ndarray of shape (n_samples, tfidf_features + embed_dim).
    """
    ...

def select_features(
    X: np.ndarray,
    y: np.ndarray,
    method: Literal["chi2", "mutual_info", "anova"] = "mutual_info",
    k: int = 5_000,
) -> tuple[np.ndarray, Any]:
    """
    Select top-k features using statistical tests.

    Returns:
        (selected_X, fitted_selector)
    """
    ...
```

### Deliverables
- [ ] TF-IDF vectorizer saved/loaded from disk correctly
- [ ] BERT embeddings extracted in batches (no OOM on 16GB GPU)
- [ ] Combined features pipeline verified on IMDB dataset

---

## Phase 3: Model Training (Days 3–5)

### Objective
Train and compare classical baseline models against fine-tuned transformer models. Build an ensemble that outperforms individual models.

### Files

#### `src/models/baseline_models.py`

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np
import joblib

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    multi_class: str = "auto",
    n_jobs: int = -1,
    save_path: Optional[str] = None,
) -> LogisticRegression:
    """
    Train Logistic Regression with L2 regularization.
    Runs 5-fold cross-validation and logs validation accuracy.

    Returns:
        Fitted LogisticRegression model.
    """
    ...

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 2,
    n_jobs: int = -1,
    save_path: Optional[str] = None,
) -> RandomForestClassifier:
    """
    Train Random Forest classifier with feature importance logging.

    Returns:
        Fitted RandomForestClassifier.
    """
    ...

def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    subsample: float = 0.8,
) -> GradientBoostingClassifier: ...

def evaluate_baseline(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute accuracy, precision, recall, F1, AUC-ROC.

    Returns:
        {
          "accuracy": float,
          "precision": float,
          "recall": float,
          "f1": float,
          "auc_roc": float,
          "classification_report": str,
        }
    """
    ...
```

#### `src/models/transformer_classifier.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    AdamW,
)
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    PyTorch Dataset for text classification.
    Handles tokenization, padding, and truncation.
    """

    def __init__(
        self,
        texts: list[str],
        labels: Optional[list[int]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None: ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: ...


class TransformerClassifier:
    """
    Wrapper for HuggingFace sequence classification models.
    Supports BERT, DistilBERT, RoBERTa, and compatible models.

    Usage:
        clf = TransformerClassifier("roberta-base", num_labels=2)
        clf.train(train_df, val_df, epochs=3, batch_size=16, lr=2e-5)
        predictions = clf.predict(["text1", "text2"])
        label, confidence = clf.predict_single("This is amazing!")
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        freeze_layers: int = 6,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name:    HuggingFace model identifier.
            num_labels:    Number of output classes.
            freeze_layers: Number of transformer layers to freeze
                           (from bottom) during fine-tuning.
                           Useful for small datasets. Set 0 to train all.
            device:        'cuda', 'cpu', or None (auto-detect).
        """
        ...

    def _freeze_layers(self, n_layers: int) -> None:
        """Freeze bottom n transformer encoder layers."""
        ...

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        warmup_ratio: float = 0.06,
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
        eval_steps: int = 100,
        save_best: bool = True,
        output_dir: str = "models/",
        text_col: str = "text",
        label_col: str = "label",
    ) -> dict:
        """
        Fine-tune the classifier using AdamW + linear warmup schedule.

        Logs training loss and validation F1 every eval_steps.
        Saves the best checkpoint by validation F1.

        Args:
            train_data:   DataFrame with 'text' and 'label' columns.
            val_data:     Validation DataFrame (same schema).
            epochs:       Number of training epochs.
            batch_size:   Training batch size.
            lr:           Peak learning rate.
            warmup_ratio: Fraction of steps for LR warmup.
            weight_decay: AdamW weight decay.
            gradient_clip: Max gradient norm for clipping.
            eval_steps:   Evaluate on validation set every N steps.
            save_best:    Save checkpoint with best val F1.
            output_dir:   Directory to save checkpoints.

        Returns:
            Training history dict with keys:
            {'train_losses', 'val_f1s', 'best_epoch', 'best_val_f1'}
        """
        ...

    def predict(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a list of texts.

        Returns:
            (labels, probabilities) — both ndarrays of shape (len(texts),).
            probabilities[i] is the confidence for the predicted label.
        """
        ...

    def predict_single(
        self,
        text: str,
    ) -> tuple[int, float]:
        """
        Classify a single text.

        Returns:
            (label: int, confidence: float)

        Example:
            >>> clf.predict_single("This movie was terrible")
            (0, 0.9731)   # 0 = negative, 97.3% confidence
        """
        ...

    def predict_proba(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Return full probability distribution over all classes.

        Returns:
            ndarray of shape (len(texts), num_labels).
        """
        ...

    def save(self, output_dir: str) -> None:
        """Save model weights, tokenizer, and config to output_dir."""
        ...

    @classmethod
    def load(cls, model_dir: str) -> "TransformerClassifier":
        """Load a saved TransformerClassifier from disk."""
        ...
```

#### `src/models/model_ensemble.py`

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
import numpy as np

class EnsembleClassifier:
    """
    Voting or stacking ensemble over multiple classifiers.

    Voting: majority vote or probability averaging.
    Stacking: meta-learner (Logistic Regression) trains on out-of-fold
              predictions from base models.

    Usage:
        ensemble = EnsembleClassifier(
            models={
                "lr": logistic_regression,
                "rf": random_forest,
                "bert": transformer_classifier,
            },
            method="soft_voting",
        )
        ensemble.fit(X_train, y_train)
        labels, probs = ensemble.predict(X_test)
    """

    def __init__(
        self,
        models: dict[str, Any],
        method: Literal["hard_voting", "soft_voting", "stacking"] = "soft_voting",
        meta_learner_C: float = 1.0,
        cv_folds: int = 5,
    ) -> None: ...

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None: ...

    def predict(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (labels, probabilities)."""
        ...

    def feature_importances(self) -> dict[str, float]:
        """Stacking only: meta-learner coefficients per base model."""
        ...
```

### Deliverables
- [ ] Logistic Regression baseline achieving ≥89% F1 on IMDB
- [ ] TransformerClassifier.train() completes without OOM on 16GB GPU
- [ ] Ensemble outperforms best single model by ≥0.5% F1

---

## Phase 4: Hyperparameter Tuning (Days 5–6)

### Objective
Use Optuna to systematically search the hyperparameter space for transformer models, reducing manual trial-and-error.

### Files

#### `src/tuning.py`

```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_hyperparameters(
    trial: optuna.Trial,
    model_type: Literal["bert", "distilbert", "roberta", "logistic", "random_forest"],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
) -> float:
    """
    Optuna objective function. Called for each trial.

    Searches over:
        For transformer models:
            - lr: [1e-6, 5e-4] (log scale)
            - batch_size: [8, 16, 32]
            - warmup_ratio: [0.0, 0.15]
            - freeze_layers: [0, 2, 4, 6, 8]
            - weight_decay: [0.0, 0.1]
        For baseline models:
            - C (LR): [0.01, 100] (log scale)
            - n_estimators (RF): [50, 500]
            - max_depth (RF): [None, 5, 10, 20]

    Returns:
        Validation F1 score (maximized by Optuna).

    Pruning:
        Reports intermediate F1 after each epoch so Optuna can prune
        unpromising trials early via HyperbandPruner.
    """
    ...


def run_optimization(
    n_trials: int = 50,
    model_type: str = "bert",
    train_data: Optional[pd.DataFrame] = None,
    val_data: Optional[pd.DataFrame] = None,
    study_name: str = "text_classification",
    storage: str = "sqlite:///optuna.db",
    n_jobs: int = 1,
    timeout_minutes: Optional[int] = 120,
) -> optuna.Study:
    """
    Create or resume an Optuna study and run optimization.

    Uses TPE sampler with Hyperband pruning for efficient search.
    Results are persisted to SQLite for resumability.

    Args:
        n_trials:         Maximum number of trials.
        model_type:       Model family to tune.
        study_name:       Name for Optuna study (for resuming).
        storage:          SQLite path for persistence.
        n_jobs:           Parallel workers (1 recommended for GPU models).
        timeout_minutes:  Stop after this many minutes regardless of trials.

    Returns:
        Completed optuna.Study object.

    Usage:
        study = run_optimization(n_trials=50, model_type="bert")
        print(f"Best params: {study.best_params}")
        print(f"Best F1: {study.best_value:.4f}")

        optuna.visualization.plot_optimization_history(study)
        optuna.visualization.plot_param_importances(study)
    """
    ...


def apply_best_params(
    study: optuna.Study,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict:
    """
    Retrain with best params on train+val, evaluate on test.

    Returns:
        Final test set metrics dict.
    """
    ...
```

### Deliverables
- [ ] Optuna study persisted to SQLite (resumable across sessions)
- [ ] Best hyperparameters logged to `logs/best_params.json`
- [ ] Visualization plots saved to `reports/hyperparameter_search.html`
- [ ] Tuned model improves baseline by ≥1% F1

---

## Phase 5: API Deployment (Days 6–8)

### Objective
Wrap the best model in a production-grade FastAPI service with input validation, error handling, authentication, and Docker containerization.

### Files

#### `src/api/models.py`

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000, description="Input text to classify")
    task: str = Field(default="sentiment", description="Classification task")
    model_version: Optional[str] = Field(default=None, description="Specific model version to use")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v.strip()


class PredictResponse(BaseModel):
    label: str
    label_id: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    model: str
    processing_time_ms: float


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    task: str = Field(default="sentiment")


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    total_time_ms: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    task: str
    num_labels: int
    label_names: list[str]
    training_f1: float
    training_date: str
```

#### `src/api/main.py`

```python
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.models import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    ModelInfoResponse,
)
from src.models.transformer_classifier import TransformerClassifier

logger = logging.getLogger(__name__)
classifier: Optional[TransformerClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global classifier
    logger.info("Loading classifier model...")
    classifier = TransformerClassifier.load("models/best_model")
    logger.info("Model loaded successfully")
    yield
    classifier = None


app = FastAPI(
    title="Text Classification API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    """Return service health status."""
    return {"status": "healthy", "model_loaded": classifier is not None}


@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    ...


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Classify a single text sample.

    Returns label, confidence score, and processing time.
    Raises HTTP 422 on invalid input, HTTP 500 on model failure.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.perf_counter()
    try:
        label_id, confidence = classifier.predict_single(request.text)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    elapsed_ms = (time.perf_counter() - start) * 1000
    return PredictResponse(
        label=classifier.label_names[label_id],
        label_id=label_id,
        confidence=float(confidence),
        model=classifier.model_name,
        processing_time_ms=elapsed_ms,
    )


@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Classify a batch of texts (max 100 per request)."""
    ...
```

#### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### API Endpoints Summary

| Method | Path | Request Body | Response |
|---|---|---|---|
| GET | `/health` | — | `{status, model_loaded}` |
| GET | `/model_info` | — | `ModelInfoResponse` |
| POST | `/predict` | `PredictRequest` | `PredictResponse` |
| POST | `/predict_batch` | `BatchPredictRequest` | `BatchPredictResponse` |

### Deliverables
- [ ] `/predict` responds in <100ms on CPU
- [ ] `/predict_batch` handles 100 texts in <1s on GPU
- [ ] Docker image builds and runs without modification
- [ ] All endpoints return proper 4xx/5xx with error detail

---

## Phase 6: Dashboard (Days 8–9)

### Objective
Build an interactive Streamlit dashboard for non-technical users to upload data, run classification, and explore results.

### Files

#### `src/dashboard/app.py`

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# Sidebar: model selection, task selection
# Main view tabs: Upload, Results, Analytics, Model Info

def render_upload_tab():
    """
    Tab 1: Upload CSV → preview → classify → download results.
    Supports batch classification with progress bar.
    """
    ...

def render_results_tab(results_df: pd.DataFrame):
    """
    Tab 2: Classified results table with color-coded confidence.
    Filter by label, search by text, sort by confidence.
    """
    ...

def render_analytics_tab(results_df: pd.DataFrame, true_labels: Optional[pd.Series]):
    """
    Tab 3: If true labels provided, show:
    - Confusion matrix (interactive Plotly heatmap)
    - Per-class precision/recall/F1 bar chart
    - Confidence distribution histogram
    - Misclassified examples browser
    """
    ...

def render_model_info_tab():
    """
    Tab 4: Model card — architecture, training data, metrics, limitations.
    """
    ...
```

### Features
- Upload CSV or paste text directly
- Real-time single-text classification with confidence gauge
- Batch classification with progress bar and download button
- Confusion matrix visualization (if ground truth labels provided)
- Per-class metrics breakdown
- Confidence distribution plot
- Misclassified examples browser

### Deliverables
- [ ] Upload CSV and classify 1000 rows in <30 seconds
- [ ] Confusion matrix renders correctly for multi-class tasks
- [ ] Export results as CSV and JSON

---

## Phase 7: Evaluation (Days 9–10)

### Objective
Comprehensive evaluation and error analysis to understand model strengths and failure modes.

### Files

#### `src/evaluation.py`

```python
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report,
)
import plotly.express as px
import plotly.figure_factory as ff
from typing import Optional

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    label_names: Optional[list[str]] = None,
    average: str = "weighted",
) -> dict:
    """
    Compute comprehensive classification metrics.

    Returns:
        {
          "accuracy": float,
          "precision": float,
          "recall": float,
          "f1": float,
          "auc_roc": float,           # None if y_proba not provided
          "per_class": {
            "label_name": {
              "precision": float,
              "recall": float,
              "f1": float,
              "support": int,
            }
          },
          "confusion_matrix": list[list[int]],
        }
    """
    ...

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Plot interactive normalized confusion matrix using Plotly.
    Values are shown as both raw counts and percentages.
    """
    ...

def error_analysis(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_names: list[str],
    n_worst: int = 50,
) -> pd.DataFrame:
    """
    Return the most confidently misclassified examples.

    Sorted by confidence descending — these are the model's "blind spots".

    Returns:
        DataFrame with columns:
        ['text', 'true_label', 'pred_label', 'confidence', 'error_type']
        where error_type is 'false_positive' or 'false_negative'.
    """
    ...

def compare_models(
    results: dict[str, dict],
    metrics: list[str] = ["accuracy", "f1", "auc_roc"],
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Grouped bar chart comparing multiple models across metrics.

    Args:
        results: {"model_name": metrics_dict, ...}
    """
    ...

def generate_report(
    model_name: str,
    metrics: dict,
    error_analysis_df: pd.DataFrame,
    output_path: str = "reports/evaluation_report.html",
) -> str:
    """
    Generate a full HTML evaluation report using Jinja2 template.

    Includes: model card, metrics table, confusion matrix, error analysis.
    Returns the path to the generated report.
    """
    ...
```

### Evaluation Criteria

| Metric | Baseline (LR) | Target (Transformer) |
|---|---|---|
| Accuracy | ≥87% | ≥94% |
| F1 (weighted) | ≥87% | ≥94% |
| AUC-ROC | ≥91% | ≥97% |
| Inference time | <5ms | <100ms |

---

## File Structure

```
06-text-classification/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py
│   │   ├── augmentor.py
│   │   └── preprocessor.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── tfidf_features.py
│   │   ├── embedding_features.py
│   │   └── feature_combiner.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_models.py
│   │   ├── transformer_classifier.py
│   │   └── model_ensemble.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── models.py
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py
│   ├── tuning.py
│   └── evaluation.py
├── tests/
│   ├── conftest.py
│   ├── test_preprocessor.py
│   ├── test_tfidf.py
│   ├── test_baseline_models.py
│   ├── test_transformer.py
│   └── test_api.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_experiments.ipynb
│   ├── 03_transformer_finetuning.ipynb
│   └── 04_error_analysis.ipynb
├── models/                   # Saved model checkpoints (gitignored)
├── data/                     # Raw and processed data (gitignored)
├── reports/                  # Evaluation reports and plots
├── logs/                     # Training logs
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── docs/
    └── PROJECT_PLAN.md
```

---

## Dependencies

```
# Core ML
transformers>=4.36.0
torch>=2.1.0
scikit-learn>=1.3.0
optuna>=3.5.0

# Data
pandas>=2.0.0
numpy>=1.24.0
datasets>=2.16.0
nltk>=3.8.0
sentencepiece>=0.1.99

# API
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0

# Dashboard & Visualization
streamlit>=1.30.0
plotly>=5.18.0

# Utilities
joblib>=1.3.0
tqdm>=4.66.0
python-dotenv>=1.0.0
loguru>=0.7.0
```

---

## Success Criteria

1. **Data pipeline**: All 4 datasets load and preprocess without errors
2. **Augmentation**: All 5 methods produce valid augmented texts
3. **Baseline models**: LR achieves ≥89% F1 on IMDB test set
4. **Transformer**: RoBERTa achieves ≥95% F1 on IMDB test set
5. **Tuning**: Optuna improves best model by ≥1% F1
6. **API**: All 4 endpoints return correct responses; /predict <100ms
7. **Docker**: `docker-compose up` runs without manual intervention
8. **Dashboard**: Classifies 1000 rows in <30 seconds
9. **Tests**: All unit tests pass (`pytest tests/ -v`)
10. **Report**: HTML evaluation report generated at `reports/`
