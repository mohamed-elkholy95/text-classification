"""FastAPI for text classification.

This module wires up the text-classification service with four endpoint
groups:

1. **Health** – lightweight liveness probe (``GET /health``).
2. **Prediction** – single & batch sentiment classification
   (``POST /predict``, ``POST /batch_predict``).
3. **Text statistics** – linguistic metrics computed inline
   (``POST /stats``).
4. **Model comparison** – side-by-side evaluation placeholder
   (``POST /compare``).

The ``/stats`` endpoint implements a simplified Flesch-Kincaid readability
formula entirely in Python (no external NLP library needed), which makes it
a good educational example of how readability scores are built from
character-level heuristics.
"""

import logging
import re
import uuid
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.models import (
    BatchInput,
    BatchPredictionResponse,
    HealthResponse,
    ModelComparisonResponse,
    PredictionResponse,
    TextInput,
    TextStatsRequest,
    TextStatsResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Text Classification API", version="1.0.0")

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
# Allow all origins in development.  In production, lock this down to the
# actual front-end domain(s) to prevent CSRF-style abuse.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request-ID middleware
# ---------------------------------------------------------------------------
# Attaching a unique ID to every request is a production best practice
# for **distributed tracing**: when something goes wrong, you can
# correlate API logs with client-side errors by matching the request ID.
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID header to every response."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)


# ---------------------------------------------------------------------------
# Global state (populated at startup by the training pipeline)
# ---------------------------------------------------------------------------
_model = None
_preprocessor = None
_tfidf = None
_classes = ["negative", "positive"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_syllables(word: str) -> int:
    """Estimate the number of syllables in *word*.

    The algorithm counts contiguous groups of vowels (a, e, i, o, u, y)
    which is a common heuristic for English text.  A minimum of 1 is
    returned for every non-empty word so that very short words (e.g. "the")
    still contribute to the readability denominator.

    This is intentionally simple — full syllabification is language-specific
    and outside the scope of a portfolio demo.

    Args:
        word: A single token (may contain punctuation).

    Returns:
        Estimated syllable count (≥ 1).
    """
    word = word.lower().strip()
    if not word:
        return 0
    # Match runs of vowel characters.  'y' is treated as a vowel here
    # because it frequently acts as one in English (e.g. "happy", "rhythm").
    vowel_groups = re.findall(r"[aeiouy]+", word)
    return max(1, len(vowel_groups))


# ---------------------------------------------------------------------------
# Endpoints – Health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health and loaded model names.

    Suitable for Kubernetes / Docker health checks.
    """
    return HealthResponse(status="healthy", models_loaded=["logistic_regression"])


# ---------------------------------------------------------------------------
# Endpoints – Prediction
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse)
async def predict(input: TextInput) -> PredictionResponse:
    """Classify a single text as *positive* or *negative*.

    When no model is loaded (e.g. in pure-API mode), a neutral default
    prediction is returned so the endpoint always responds without error.

    The endpoint performs basic input sanitization: texts consisting
    entirely of whitespace are rejected with a 422 error rather than
    producing meaningless predictions.

    Args:
        input: JSON body with a ``text`` field.

    Returns:
        :class:`PredictionResponse` with label, class name, and confidence.

    Raises:
        HTTPException(422): If text is empty or whitespace-only.
    """
    # Guard against whitespace-only inputs that would produce meaningless
    # TF-IDF vectors (all zeros) and therefore arbitrary predictions.
    if not input.text.strip():
        raise HTTPException(
            status_code=422,
            detail="Text must contain at least one non-whitespace character.",
        )
    if _tfidf is None:
        return PredictionResponse(
            text=input.text,
            predicted_label=0,
            predicted_class="negative",
            confidence=0.5,
        )

    features = _tfidf.transform([input.text])
    pred = _model.predict(features)[0]
    proba = _model.predict_proba(features)[0]
    label = int(pred)
    return PredictionResponse(
        text=input.text,
        predicted_label=label,
        predicted_class=_classes[label] if label < len(_classes) else str(label),
        confidence=round(float(max(proba)), 4),
    )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(input: BatchInput) -> BatchPredictionResponse:
    """Classify multiple texts in a single request.

    Batch inference is more efficient than calling ``/predict`` repeatedly
    because the TF-IDF vectoriser processes the entire list at once.

    Args:
        input: JSON body with a ``texts`` list.

    Returns:
        :class:`BatchPredictionResponse` with an ordered list of predictions.
    """
    if _tfidf is None:
        preds = [
            PredictionResponse(
                text=t, predicted_label=0,
                predicted_class="negative", confidence=0.5,
            )
            for t in input.texts
        ]
        return BatchPredictionResponse(predictions=preds, total=len(preds))

    features = _tfidf.transform(input.texts)
    labels = _model.predict(features)
    probas = _model.predict_proba(features)
    preds = [
        PredictionResponse(
            text=text,
            predicted_label=int(l),
            predicted_class=_classes[int(l)] if int(l) < len(_classes) else str(l),
            confidence=round(float(max(p)), 4),
        )
        for text, l, p in zip(input.texts, labels, probas)
    ]
    return BatchPredictionResponse(predictions=preds, total=len(preds))


# ---------------------------------------------------------------------------
# Endpoints – Text statistics
# ---------------------------------------------------------------------------


@app.post("/stats", response_model=TextStatsResponse)
async def compute_text_stats(input: TextStatsRequest) -> TextStatsResponse:
    """Compute linguistic statistics for the supplied text.

    The endpoint demonstrates common NLP feature-engineering techniques
    (token counting, Type-Token Ratio, Flesch-Kincaid readability) that
    would typically feed into a classifier or be surfaced to end-users.

    **Flesch-Kincaid formula (simplified):**

    ``206.835 − 1.015 × (words / sentences) − 84.6 × (syllables / words)``

    Scores range roughly from 0 (very difficult) to 100 (very easy).

    Args:
        input: JSON body with a ``text`` field.

    Returns:
        :class:`TextStatsResponse` with word count, sentence count, averages,
        vocabulary richness, and readability score.
    """
    text: str = input.text

    # --- Word-level metrics ---
    # ``str.split()`` splits on any whitespace, which is good enough for
    # a demo.  A production pipeline might use ``nltk.word_tokenize()`` or
    # spaCy's tokenizer to handle punctuation correctly.
    words: List[str] = text.split()
    total_words: int = len(words)

    # Guard against empty input to avoid division-by-zero.
    if total_words == 0:
        return TextStatsResponse(
            word_count=0,
            sentence_count=0,
            avg_word_length=0.0,
            avg_sentence_length=0.0,
            vocabulary_richness=0.0,
            readability_score=0.0,
        )

    # --- Sentence-level metrics ---
    # Split on sentence-ending punctuation.  The regex keeps the empty
    # trailing string produced by a final period, so we filter it out.
    sentences: List[str] = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    total_sentences: int = max(len(sentences), 1)  # avoid 0-denominator

    # --- Averages ---
    # Average word length is a classic readability proxy — longer words
    # generally indicate more complex or technical writing.
    avg_word_length: float = round(
        sum(len(w) for w in words) / total_words, 4
    )
    avg_sentence_length: float = round(total_words / total_sentences, 4)

    # --- Vocabulary richness (Type-Token Ratio) ---
    # TTR = unique_words / total_words.  High TTR (close to 1) means
    # the author uses many different words; low TTR suggests repetition.
    unique_words = set(w.lower() for w in words)
    vocabulary_richness: float = round(len(unique_words) / total_words, 4)

    # --- Readability (simplified Flesch-Kincaid) ---
    # Syllable estimation uses vowel-group counting (see helper above).
    total_syllables: int = sum(_count_syllables(w) for w in words)
    readability_score: float = round(
        206.835
        - 1.015 * (total_words / total_sentences)
        - 84.6 * (total_syllables / total_words),
        4,
    )

    return TextStatsResponse(
        word_count=total_words,
        sentence_count=len(sentences),
        avg_word_length=avg_word_length,
        avg_sentence_length=avg_sentence_length,
        vocabulary_richness=vocabulary_richness,
        readability_score=readability_score,
    )


# ---------------------------------------------------------------------------
# Endpoints – Model comparison
# ---------------------------------------------------------------------------


@app.post("/compare", response_model=ModelComparisonResponse)
async def compare_models() -> ModelComparisonResponse:
    """Return a placeholder comparison of trained model performances.

    In a real system this endpoint would query a results database (e.g.
    MLflow, Weights & Biases) and aggregate metrics across experiments.
    For this portfolio demo we return hard-coded representative numbers
    so the API contract is fully exercisable.

    Returns:
        :class:`ModelComparisonResponse` with per-model metrics and the
        name of the best-scoring model.
    """
    # Placeholder data representing typical sentiment-analysis results.
    # These numbers are illustrative — they are **not** from actual training.
    models: List[dict] = [
        {
            "name": "logistic_regression",
            "accuracy": 0.892,
            "f1_score": 0.887,
            "train_time_s": 1.2,
        },
        {
            "name": "svm_linear",
            "accuracy": 0.901,
            "f1_score": 0.895,
            "train_time_s": 3.4,
        },
        {
            "name": "random_forest",
            "accuracy": 0.856,
            "f1_score": 0.843,
            "train_time_s": 8.7,
        },
    ]

    # Pick the best model by F1 score (commonly preferred over accuracy
    # for imbalanced datasets).
    best_model: str = max(models, key=lambda m: m["f1_score"])["name"]

    return ModelComparisonResponse(models=models, best_model=best_model)


# ---------------------------------------------------------------------------
# Entrypoint (for local development)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
