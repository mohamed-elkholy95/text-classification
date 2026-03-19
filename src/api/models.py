"""Pydantic models for the text classification API."""
from typing import List, Optional
from pydantic import BaseModel, Field


class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class BatchInput(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    text: str
    predicted_label: int
    predicted_class: str
    confidence: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int


class HealthResponse(BaseModel):
    status: str = "healthy"
    models_loaded: List[str] = []
