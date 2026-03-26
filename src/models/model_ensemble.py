"""Model ensemble for text classification.

Ensemble Methods — Why Combining Models Works:
───────────────────────────────────────────────
The idea behind ensembling is that **diverse models make different
errors**.  If one model fails on a particular input, the others are
likely to get it right — and the majority vote (or averaged probability)
cancels out individual mistakes.

There are two main voting strategies:

• **Hard voting** — each model casts one vote (its predicted class).
  The class with the most votes wins.  Simple, robust, but ignores
  model confidence.

• **Soft voting** — each model contributes its predicted probability
  distribution.  Probabilities are averaged (optionally weighted), and
  the class with the highest average probability wins.  Soft voting
  usually outperforms hard voting because it uses more information.

**Weighted voting** assigns different importance to each model.  A
well-calibrated model might get weight 2.0 while a weaker baseline
gets 0.5.  Weights can be set manually or derived from validation
performance (e.g., weight = validation F1 score).

References:
    Dietterich, T. G. (2000). *Ensemble Methods in Machine Learning*.
    Multiple Classifier Systems, LNCS 1857, pp. 1–15.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """Voting ensemble of classification models.

    Supports hard voting (majority label), soft voting (averaged
    probabilities), and optional per-model weights.

    Example::

        >>> ensemble = ModelEnsemble(strategy="soft")
        >>> ensemble.add_model("lr", lr_model, weight=2.0)
        >>> ensemble.add_model("nb", nb_model, weight=1.0)
        >>> predictions = ensemble.predict(X_test)
    """

    def __init__(self, strategy: str = "soft") -> None:
        if strategy not in ("soft", "hard"):
            raise ValueError(f"strategy must be 'soft' or 'hard'; got '{strategy}'.")
        self._models: Dict[str, object] = {}
        self._weights: Dict[str, float] = {}
        self._strategy = strategy

    def add_model(self, name: str, model: object, weight: float = 1.0) -> None:
        """Register a model with an optional weight.

        Args:
            name: Model identifier (must be unique).
            model: Fitted model with a ``predict`` method.
            weight: Importance weight for voting.  Higher weights give
                this model more influence.  Defaults to 1.0.

        Raises:
            ValueError: If *weight* is not positive.
        """
        if weight <= 0:
            raise ValueError(f"Weight must be positive; got {weight}.")
        self._models[name] = model
        self._weights[name] = weight
        logger.info("Added model '%s' to ensemble (weight=%.2f)", name, weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction.

        Args:
            X: Feature matrix.

        Returns:
            Predicted labels.
        """
        if not self._models:
            raise RuntimeError("No models registered")

        predictions = {}
        for name, model in self._models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as exc:
                logger.warning("Model '%s' predict failed: %s", name, exc)

        if self._strategy == "soft":
            return self._soft_vote(predictions)
        return self._hard_vote(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Ensemble probability prediction with optional weighting.

        Each model's probability distribution is scaled by its weight
        before averaging.  This lets higher-quality models contribute
        more to the final prediction.

        For models without ``predict_proba`` (e.g., ``LinearSVC``),
        the ``decision_function`` output is converted to pseudo-
        probabilities via softmax normalization.

        Args:
            X: Feature matrix.

        Returns:
            Weighted-average probability matrix (n_samples, n_classes).
        """
        probas = []
        weights = []
        for name, model in self._models.items():
            try:
                if hasattr(model, "predict_proba"):
                    probas.append(model.predict_proba(X))
                    weights.append(self._weights[name])
                elif hasattr(model, "decision_function"):
                    df = model.decision_function(X)
                    if df.ndim == 1:
                        # Binary case: decision_function returns 1-D
                        df = np.column_stack([-df, df])
                    # Softmax to convert decision scores to probabilities
                    exp_df = np.exp(df - df.max(axis=1, keepdims=True))
                    probas.append(exp_df / exp_df.sum(axis=1, keepdims=True))
                    weights.append(self._weights[name])
            except Exception as exc:
                logger.warning("Model '%s' predict_proba failed: %s", name, exc)

        if not probas:
            return np.zeros((X.shape[0], 2))

        # Weighted average: sum(weight_i * proba_i) / sum(weight_i)
        weights_arr = np.array(weights)
        weighted_sum = sum(w * p for w, p in zip(weights_arr, probas))
        return weighted_sum / weights_arr.sum()

    def _soft_vote(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted soft voting."""
        probas = self.predict_proba(np.zeros((len(next(iter(predictions.values()))), 1)))
        return np.argmax(probas, axis=1)

    def _hard_vote(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted majority hard voting.

        Each model's vote is counted ``weight`` times (rounded to int).
        With equal weights this reduces to standard majority voting.
        """
        n_samples = len(next(iter(predictions.values())))
        # Determine the number of classes from predictions
        all_preds = np.concatenate(list(predictions.values()))
        n_classes = int(all_preds.max()) + 1

        # Accumulate weighted votes
        votes = np.zeros((n_samples, n_classes), dtype=float)
        for name, preds in predictions.items():
            w = self._weights.get(name, 1.0)
            for i, p in enumerate(preds):
                votes[i, int(p)] += w

        return np.argmax(votes, axis=1)

    @property
    def model_names(self) -> List[str]:
        return list(self._models.keys())
