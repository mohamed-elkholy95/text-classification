"""End-to-end prediction pipeline for text classification.

Why a Pipeline Class?
─────────────────────
In production, inference involves multiple steps: clean the text,
extract features, run the model, and format the output.  Scattering
these steps across multiple files makes deployment error-prone — it's
easy to forget a preprocessing step or apply a different TF-IDF
configuration than what the model was trained with.

A **pipeline** bundles all stages into a single object with a clean
``predict()`` interface.  This ensures:

1. **Consistency** — the same preprocessing runs at training and
   inference time (training-serving skew is a top-5 ML bug).
2. **Simplicity** — API code calls ``pipeline.predict(text)`` instead
   of manually orchestrating four modules.
3. **Testability** — one object to mock in unit tests.
4. **Serializability** — save/load the entire pipeline as one artifact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.data.preprocessor import TextPreprocessor
from src.features.tfidf_features import TfidfFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured prediction output for a single text.

    Attributes:
        text: The original (uncleaned) input text.
        label: Predicted integer class label.
        class_name: Human-readable class name (if available).
        confidence: Probability of the predicted class.
        probabilities: Full probability distribution over all classes.
    """

    text: str
    label: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)


class TextClassificationPipeline:
    """End-to-end pipeline: preprocess → featurize → predict.

    Wraps the preprocessor, feature extractor, and classifier into a
    single callable object suitable for serving via API or CLI.

    Example::

        >>> pipeline = TextClassificationPipeline(
        ...     preprocessor=preprocessor,
        ...     feature_extractor=tfidf,
        ...     model=lr_model,
        ...     class_names=["negative", "positive"],
        ... )
        >>> result = pipeline.predict("This product is amazing!")
        >>> result.class_name
        'positive'
        >>> result.confidence > 0.5
        True
    """

    def __init__(
        self,
        preprocessor: TextPreprocessor,
        feature_extractor: TfidfFeatureExtractor,
        model: Any,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self._preprocessor = preprocessor
        self._feature_extractor = feature_extractor
        self._model = model
        self._class_names = class_names or []

    def predict(self, text: str) -> PredictionResult:
        """Classify a single text through the full pipeline.

        Steps:
        1. Clean the text (lowercase, remove URLs/mentions/special chars)
        2. Transform to TF-IDF features using the fitted vectorizer
        3. Run the classifier to get label and probabilities
        4. Package into a PredictionResult

        Args:
            text: Raw input text (any length, any formatting).

        Returns:
            PredictionResult with label, class name, confidence, and
            full probability distribution.
        """
        cleaned = self._preprocessor.clean_text(text)
        features = self._feature_extractor.transform([cleaned])
        label = int(self._model.predict(features)[0])

        # Extract probabilities if the model supports them
        probabilities: Dict[str, float] = {}
        confidence = 1.0
        if hasattr(self._model, "predict_proba"):
            try:
                proba = self._model.predict_proba(features)[0]
                confidence = float(proba.max())
                for i, p in enumerate(proba):
                    name = self._class_names[i] if i < len(self._class_names) else str(i)
                    probabilities[name] = round(float(p), 4)
            except Exception as exc:
                logger.warning("predict_proba failed: %s", exc)

        class_name = (
            self._class_names[label]
            if label < len(self._class_names)
            else str(label)
        )

        return PredictionResult(
            text=text,
            label=label,
            class_name=class_name,
            confidence=round(confidence, 4),
            probabilities=probabilities,
        )

    def predict_batch(self, texts: Sequence[str]) -> List[PredictionResult]:
        """Classify multiple texts efficiently.

        Batch inference is faster than calling ``predict()`` in a loop
        because the TF-IDF vectorizer processes all texts at once and
        the model can exploit vectorized operations.

        Args:
            texts: Sequence of raw input texts.

        Returns:
            List of PredictionResult objects in the same order as input.
        """
        if not texts:
            return []

        cleaned = [self._preprocessor.clean_text(t) for t in texts]
        features = self._feature_extractor.transform(cleaned)
        labels = self._model.predict(features)

        # Batch probability extraction
        probas = None
        if hasattr(self._model, "predict_proba"):
            try:
                probas = self._model.predict_proba(features)
            except Exception:
                pass

        results = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            label = int(label)
            probabilities: Dict[str, float] = {}
            confidence = 1.0

            if probas is not None:
                confidence = float(probas[i].max())
                for j, p in enumerate(probas[i]):
                    name = self._class_names[j] if j < len(self._class_names) else str(j)
                    probabilities[name] = round(float(p), 4)

            class_name = (
                self._class_names[label]
                if label < len(self._class_names)
                else str(label)
            )

            results.append(PredictionResult(
                text=text,
                label=label,
                class_name=class_name,
                confidence=round(confidence, 4),
                probabilities=probabilities,
            ))

        logger.info("Batch prediction: %d texts classified.", len(results))
        return results
