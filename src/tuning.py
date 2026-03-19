"""Hyperparameter tuning with Optuna (mock fallback)."""
import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.info("optuna not installed")


class OptunaOptimizer:
    """Optuna-based hyperparameter optimization."""

    def __init__(self, model_type: str = "logistic_regression", n_trials: int = 20, seed: int = 42) -> None:
        self.model_type = model_type
        self.n_trials = n_trials
        self.seed = seed
        self._best_params: Optional[Dict] = None

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.

        Returns:
            Best parameters found.
        """
        if not OPTUNA_AVAILABLE:
            logger.info("Optuna unavailable — returning defaults")
            return self._default_params()

        def objective(trial):
            params = self._suggest_params(trial)
            model = self._build_model(params)
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            return score

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self._best_params = study.best_params
        logger.info("Best params: %s (score=%.4f)", study.best_params, study.best_value)
        return study.best_params

    def _suggest_params(self, trial) -> Dict:
        if self.model_type == "logistic_regression":
            return {"C": trial.suggest_float("C", 0.01, 100, log=True)}
        elif self.model_type == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
            }
        elif self.model_type == "svm":
            return {"C": trial.suggest_float("C", 0.01, 100, log=True)}
        return {"C": 1.0}

    def _build_model(self, params: Dict) -> Any:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import LinearSVC

        if self.model_type == "logistic_regression":
            return LogisticRegression(C=params["C"], max_iter=1000, random_state=self.seed)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(random_state=self.seed, **params)
        elif self.model_type == "svm":
            return LinearSVC(C=params["C"], random_state=self.seed, max_iter=10000)
        return LogisticRegression(C=1.0, random_state=self.seed)

    def _default_params(self) -> Dict:
        return {"C": 1.0}
