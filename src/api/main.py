"""FastAPI for text classification."""
import logging
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import TextInput, BatchInput, PredictionResponse, BatchPredictionResponse, HealthResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Text Classification API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None
_preprocessor = None
_tfidf = None
_classes = ["negative", "positive"]


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", models_loaded=["logistic_regression"])


@app.post("/predict", response_model=PredictionResponse)
async def predict(input: TextInput):
    """Classify a single text."""
    if _tfidf is None:
        return PredictionResponse(
            text=input.text, predicted_label=0,
            predicted_class="negative", confidence=0.5,
        )
    features = _tfidf.transform([input.text])
    pred = _model.predict(features)[0]
    proba = _model.predict_proba(features)[0]
    label = int(pred)
    return PredictionResponse(
        text=input.text, predicted_label=label,
        predicted_class=_classes[label] if label < len(_classes) else str(label),
        confidence=round(float(max(proba)), 4),
    )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(input: BatchInput):
    """Classify multiple texts."""
    if _tfidf is None:
        preds = [PredictionResponse(text=t, predicted_label=0, predicted_class="negative", confidence=0.5)
                 for t in input.texts]
        return BatchPredictionResponse(predictions=preds, total=len(preds))

    features = _tfidf.transform(input.texts)
    labels = _model.predict(features)
    probas = _model.predict_proba(features)
    preds = [
        PredictionResponse(text=text, predicted_label=int(l),
                           predicted_class=_classes[int(l)] if int(l) < len(_classes) else str(l),
                           confidence=round(float(max(p)), 4))
        for text, l, p in zip(input.texts, labels, probas)
    ]
    return BatchPredictionResponse(predictions=preds, total=len(preds))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
