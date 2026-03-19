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
