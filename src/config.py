"""Text Classification Pipeline — Configuration."""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MAX_VOCAB_SIZE = 50000
MAX_SEQUENCE_LENGTH = 512

MODELS_CONFIG = {
    "logistic_regression": {"C": 1.0, "max_iter": 1000},
    "naive_bayes": {"alpha": 1.0},
    "svm": {"C": 1.0, "kernel": "linear"},
    "random_forest": {"n_estimators": 200, "max_depth": 10},
}

TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "max_features": 20000,
    "min_df": 1,
    "max_df": 1.0,
}

API_HOST = "0.0.0.0"
API_PORT = 8002

STREAMLIT_THEME = {
    "primaryColor": "#1f77b4",
    "backgroundColor": "#0e1117",
    "secondaryBackgroundColor": "#262730",
    "textColor": "#ffffff",
}

# ─── Calibration ─────────────────────────────────────────────────────────────
# Post-hoc calibration transforms raw decision scores into well-calibrated
# probabilities so that "70 % confident" really means ~70 % accuracy.
# See src/calibration.py for detailed rationale.
CALIBRATION_CONFIG = {
    # "platt" (logistic sigmoid) is fast and works well with small
    # calibration sets; "isotonic" is more flexible but needs more data.
    "method": "platt",
}

# ─── Cross-Validation ────────────────────────────────────────────────────────
# Stratified k-fold CV provides a lower-variance estimate of model
# performance than a single split, and stratification preserves class
# proportions in each fold — critical for imbalanced datasets.
# See src/cross_validation.py for details.
CROSS_VALIDATION_CONFIG = {
    # 5 folds is the most common default: it uses 80 % for training
    # and gives a reasonable bias-variance trade-off in the estimate.
    "n_splits": 5,
    "random_state": RANDOM_SEED,
    "shuffle": True,
}

# ─── Learning Curves ─────────────────────────────────────────────────────────
# Learning curves diagnose underfitting vs overfitting by plotting
# performance as training-set size grows.  The fractions below define
# the points at which we measure the curve.
# See src/learning_curves.py for the bias–variance interpretation.
LEARNING_CURVE_CONFIG = {
    # Fractions of the training data to evaluate at.  Starting small
    # (10 %) highlights high-bias issues; ending at 100 % shows the
    # best achievable performance with the current dataset.
    "train_fractions": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    "n_repeats": 1,
    "random_state": RANDOM_SEED,
}

# ─── Text Analysis ───────────────────────────────────────────────────────────
# Corpus-level text statistics used in EDA and feature engineering.
# Understanding avg word count, vocabulary richness, and readability
# per class can reveal spurious correlations or missing features.
# See src/text_analyzer.py for implementation.
TEXT_ANALYSIS_CONFIG = {
    # Number of top words to report in the frequency table.
    # 20–50 is a good range for quick EDA; increase for deeper analysis.
    "top_n_words": 20,
}
