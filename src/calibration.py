"""Confidence calibration for binary classification models.

Why Calibration Matters:
────────────────────────
Most classifiers output a decision score (e.g., SVM margin, logistic
logit) that is *not* guaranteed to be a well-calibrated probability.
A model that predicts "70 % confident" should, across many such
predictions, be correct roughly 70 % of the time.  When it isn't,
downstream decisions (threshold tuning, risk assessment, multi-model
ensembling) become unreliable.

A *reliability diagram* visualises calibration by binning predictions
by confidence and plotting the actual frequency of the positive class.
A perfectly calibrated model's curve hugs the diagonal y = x line.

Platt Scaling and Isotonic Regression are two post-hoc methods that
learn a monotonic mapping from raw scores to calibrated probabilities:

• **Platt Scaling** (1999) fits a logistic sigmoid to the scores.
  Simple, fast, but may underfit complex miscalibrations.  Works well
  when the dataset is small or the base model is already roughly
  calibrated.

• **Isotonic Regression** (Zadrozny & Elkan, 2002) fits a
  non-parametric, piecewise-constant, non-decreasing function to the
  scores.  More flexible than Platt — can capture arbitrary monotonic
  distortions — but requires more data to avoid overfitting.

Rule of thumb: use **Platt** when you have ≤ 1 000 calibration samples,
**Isotonic** when you have more.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


class Calibrator:
    """Post-hoc probability calibration for binary classifiers.

    Wraps either Platt Scaling (parametric sigmoid) or Isotonic
    Regression (non-parametric monotone) so that predicted probabilities
    better reflect true likelihoods.

    Attributes:
        method: Calibration method — ``"platt"`` or ``"isotonic"``.
        model: The fitted calibration model (LogisticRegression or
            IsotonicRegression).

    Example::

        >>> from sklearn.svm import SVC
        >>> from src.calibration import Calibrator
        >>> clf = SVC().fit(X_train, y_train)
        >>> scores = clf.decision_function(X_valid)
        >>> cal = Calibrator(method="platt").fit(scores, y_valid)
        >>> probs = cal.predict_proba(clf.decision_function(X_test))
        >>> round(float(probs[0]), 4)
        0.8431
    """

    def __init__(self, method: str = "platt") -> None:
        """Initialise the calibrator.

        Args:
            method: ``"platt"`` for logistic sigmoid mapping,
                ``"isotonic"`` for isotonic regression.  Defaults to
                ``"platt"``.

        Raises:
            ValueError: If *method* is not one of the supported values.
        """
        if method not in ("platt", "isotonic"):
            raise ValueError(
                f"Unsupported calibration method '{method}'. "
                "Choose 'platt' or 'isotonic'."
            )
        self.method: str = method
        self.model: Optional[LogisticRegression | IsotonicRegression] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
    ) -> "Calibrator":
        """Fit the calibration model on raw decision scores.

        Args:
            scores: 1-D array of raw decision-function scores (e.g.
                ``clf.decision_function(X)``).
            y_true: 1-D array of ground-truth binary labels (0/1).

        Returns:
            ``self``, for method chaining.

        Raises:
            ValueError: If *scores* and *y_true* have incompatible
                shapes or contain unexpected values.
        """
        scores = np.asarray(scores, dtype=np.float64).ravel()
        y_true = np.asarray(y_true, dtype=np.float64).ravel()

        self._validate_inputs(scores, y_true)

        if self.method == "platt":
            # Platt Scaling: a LogisticRegression with no intercept
            # penalty applied to the raw scores.  The learned sigmoid
            # maps scores → [0, 1].
            #
            # We suppress the intercept because decision-function scores
            # are already centred around zero for many models (SVM,
            # logistic regression logit), so adding another intercept
            # would be redundant and could hurt calibration.
            self.model = LogisticRegression(
                C=1e10,  # effectively unpenalised
                fit_intercept=True,
                solver="lbfgs",
                max_iter=1000,
            )
            self.model.fit(scores.reshape(-1, 1), y_true)
            logger.info("Platt Scaling fitted on %d samples.", len(y_true))

        else:
            # Isotonic Regression: finds the least-squares
            # non-decreasing step function from scores → probabilities.
            # Because isotonic regression has no parametric assumptions
            # it can capture arbitrarily complex calibration curves,
            # at the cost of requiring more data.
            self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.model.fit(scores, y_true)
            logger.info("Isotonic Regression fitted on %d samples.", len(y_true))

        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities.

        Args:
            scores: 1-D array of raw decision-function scores.

        Returns:
            1-D array of calibrated probabilities in [0, 1].

        Raises:
            RuntimeError: If the calibrator has not been fitted yet.
            ValueError: If *scores* is empty.
        """
        if self.model is None:
            raise RuntimeError("Calibrator has not been fitted. Call `.fit()` first.")

        scores = np.asarray(scores, dtype=np.float64).ravel()
        if scores.size == 0:
            raise ValueError("scores must not be empty.")

        if self.method == "platt":
            # LogisticRegression.predict_proba returns shape (n, 2);
            # we want the probability of the positive class (column 1).
            return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]
        else:
            # IsotonicRegression.predict returns 1-D directly.
            return self.model.predict(scores)

    def score(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Compute the Brier score loss for calibrated probabilities.

        The **Brier score** is the mean squared error between predicted
        probabilities and ground-truth binary labels.  Lower is better
        (0 = perfect, 0.25 = random for balanced classes).

        Args:
            scores: 1-D array of raw decision-function scores.
            y_true: 1-D array of ground-truth binary labels (0/1).

        Returns:
            Brier score loss as a float.

        Example::

            >>> loss = cal.score(scores_valid, y_valid)
            >>> round(loss, 4)
            0.1823
        """
        probs = self.predict_proba(scores)
        return brier_score_loss(y_true, probs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(scores: np.ndarray, y_true: np.ndarray) -> None:
        """Raise ``ValueError`` if inputs are inconsistent."""
        if scores.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} vs y_true {y_true.shape}."
            )
        unique_labels = set(np.unique(y_true))
        if not unique_labels.issubset({0.0, 1.0}):
            raise ValueError(
                f"y_true must contain only 0 and 1; found {unique_labels}."
            )


def calibrate_predictions(
    scores: np.ndarray,
    y_true: np.ndarray,
    method: str = "platt",
) -> Dict[str, object]:
    """End-to-end convenience function: fit, predict, and report.

    This is a thin wrapper around :class:`Calibrator` intended for
    quick experimentation or notebook use.  For fine-grained control
    (e.g., fitting on a validation set, applying to a test set),
    instantiate :class:`Calibrator` directly.

    Args:
        scores: 1-D array of raw decision-function scores.
        y_true: 1-D array of ground-truth binary labels (0/1).
        method: ``"platt"`` or ``"isotonic"``.

    Returns:
        Dictionary containing:
        - ``"probabilities"``: calibrated probability array
        - ``"brier_score"``: Brier score loss (float)
        - ``"method"``: calibration method used (str)

    Example::

        >>> from src.calibration import calibrate_predictions
        >>> result = calibrate_predictions(scores_valid, y_valid, method="isotonic")
        >>> result["brier_score"] < 0.20
        True
    """
    calibrator = Calibrator(method=method)
    calibrator.fit(scores, y_true)
    probs = calibrator.predict_proba(scores)
    brier = brier_score_loss(y_true, probs)

    logger.info(
        "Calibration (%s) complete — Brier score: %.4f", method, brier
    )

    return {
        "probabilities": probs,
        "brier_score": float(brier),
        "method": method,
    }
