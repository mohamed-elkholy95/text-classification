"""Pydantic models for the text classification API.

Pydantic models serve as the contract between API consumers and the server.
They validate incoming request bodies and serialize outgoing responses,
ensuring type safety and automatic OpenAPI documentation generation.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TextInput(BaseModel):
    """Request model for single-text classification.

    Args:
        text: The input text to classify. Constrained to 1–10 000 characters
            to prevent abuse while supporting reasonable paragraph lengths.
    """

    text: str = Field(..., min_length=1, max_length=10000)


class BatchInput(BaseModel):
    """Request model for batch classification.

    Args:
        texts: A list of texts to classify. Limited to 100 items per request
            to keep latency bounded.
    """

    texts: List[str] = Field(..., min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    """Response model containing a single prediction result.

    Attributes:
        text: The original input text (echoed back for traceability).
        predicted_label: The integer class label produced by the model.
        predicted_class: Human-readable class name (e.g. "positive").
        confidence: Probability of the predicted class, rounded to 4 decimals.
    """

    text: str
    predicted_label: int
    predicted_class: str
    confidence: float


class BatchPredictionResponse(BaseModel):
    """Response model wrapping a list of individual predictions.

    Attributes:
        predictions: Ordered list of PredictionResponse objects.
        total: Total number of predictions returned (mirrors input count).
    """

    predictions: List[PredictionResponse]
    total: int


class HealthResponse(BaseModel):
    """Liveness / readiness probe response.

    Attributes:
        status: Should be ``"healthy"`` when the service is operational.
        models_loaded: Names of ML models currently held in memory.
    """

    status: str = "healthy"
    models_loaded: List[str] = []


class TextStatsRequest(BaseModel):
    """Request body for the ``POST /stats`` endpoint.

    Args:
        text: The raw text whose linguistic statistics should be computed.
    """

    text: str


class TextStatsResponse(BaseModel):
    """Response containing computed text statistics.

    These metrics are commonly used in NLP feature engineering and
    readability analysis. Exposing them via API lets front-end
    applications display rich insights alongside classification results.

    Attributes:
        word_count: Total number of whitespace-separated tokens.
        sentence_count: Number of sentences (split on ``.!?`` boundaries).
        avg_word_length: Mean character count per word.
        avg_sentence_length: Mean word count per sentence.
        vocabulary_richness: Ratio of unique words to total words
            (Type-Token Ratio). Values closer to 1 indicate diverse vocabulary.
        readability_score: Simplified Flesch-Kincaid readability index
            (0–100 scale; higher = easier to read).
    """

    word_count: int
    sentence_count: int
    avg_word_length: float
    avg_sentence_length: float
    vocabulary_richness: float
    readability_score: float


class ModelComparisonResponse(BaseModel):
    """Response for side-by-side model performance comparison.

    Useful for demonstrating portfolio-level evaluation skills by
    surfacing accuracy / F1 metrics across multiple classifiers.

    Attributes:
        models: List of dicts, each describing one model's metrics
            (name, accuracy, f1_score, etc.).
        best_model: Name of the highest-scoring model.
    """

    models: List[Dict[str, Any]]
    best_model: str
