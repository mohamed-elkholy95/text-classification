"""Model persistence — save and load trained pipelines.

Why Persistence Matters:
────────────────────────
Training a model can take minutes to hours.  Once trained, you want to
**serialize** (save) the model so it can be loaded instantly for
inference without retraining.  This is essential for:

• **Production deployment** — load a pre-trained model at API startup.
• **Reproducibility** — save the exact model that produced a result set.
• **Experiment tracking** — compare saved models across experiments.

Python's ``joblib`` library is the standard choice for scikit-learn
models because it handles NumPy arrays and sparse matrices more
efficiently than ``pickle``.  For safety, we also save metadata (model
class name, feature count, creation timestamp) alongside the model so
that loading code can verify compatibility.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.config import MODEL_DIR

logger = logging.getLogger(__name__)


def save_model(
    model: Any,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    directory: Optional[Path] = None,
) -> Path:
    """Serialize a trained model to disk with companion metadata.

    The model is saved as a ``.joblib`` file and an accompanying
    ``.meta.json`` file records provenance information (class name,
    timestamp, custom metadata).

    Args:
        model: A fitted scikit-learn estimator or pipeline.
        name: Base filename (without extension) for the saved artifacts.
        metadata: Optional dictionary of extra metadata to persist
            (e.g., hyperparameters, dataset name, evaluation metrics).
        directory: Target directory.  Defaults to ``config.MODEL_DIR``.

    Returns:
        Path to the saved ``.joblib`` file.

    Raises:
        ValueError: If *name* is empty.

    Example::

        >>> from sklearn.linear_model import LogisticRegression
        >>> model = LogisticRegression().fit(X_train, y_train)
        >>> path = save_model(model, "lr_v1", metadata={"C": 1.0})
        >>> path.suffix
        '.joblib'
    """
    if not name:
        raise ValueError("Model name must not be empty.")

    directory = Path(directory) if directory else MODEL_DIR
    directory.mkdir(parents=True, exist_ok=True)

    model_path = directory / f"{name}.joblib"
    meta_path = directory / f"{name}.meta.json"

    # Serialize the model using joblib (efficient for numpy arrays)
    joblib.dump(model, model_path)

    # Build provenance metadata
    meta: Dict[str, Any] = {
        "model_class": type(model).__name__,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "joblib_file": model_path.name,
    }
    if metadata:
        meta["custom"] = metadata

    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("Model saved → %s (meta → %s)", model_path, meta_path)
    return model_path


def load_model(
    name: str,
    directory: Optional[Path] = None,
) -> Any:
    """Deserialize a previously saved model.

    Args:
        name: Base filename (without extension) used when saving.
        directory: Source directory.  Defaults to ``config.MODEL_DIR``.

    Returns:
        The deserialized model object.

    Raises:
        FileNotFoundError: If the ``.joblib`` file does not exist.

    Example::

        >>> model = load_model("lr_v1")
        >>> model.predict(X_test)
        array([0, 1, 1, ...])
    """
    directory = Path(directory) if directory else MODEL_DIR
    model_path = directory / f"{name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_path}. "
            f"Available models: {list_saved_models(directory)}"
        )

    model = joblib.load(model_path)
    logger.info("Model loaded ← %s (%s)", model_path, type(model).__name__)
    return model


def load_metadata(
    name: str,
    directory: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load the metadata JSON for a saved model.

    Args:
        name: Base filename (without extension).
        directory: Source directory.  Defaults to ``config.MODEL_DIR``.

    Returns:
        Metadata dictionary.

    Raises:
        FileNotFoundError: If the ``.meta.json`` file does not exist.
    """
    directory = Path(directory) if directory else MODEL_DIR
    meta_path = directory / f"{name}.meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata file at {meta_path}.")

    return json.loads(meta_path.read_text())


def list_saved_models(directory: Optional[Path] = None) -> list[str]:
    """Return the names of all saved models in the directory.

    Args:
        directory: Directory to scan.  Defaults to ``config.MODEL_DIR``.

    Returns:
        List of model base names (without the ``.joblib`` extension).
    """
    directory = Path(directory) if directory else MODEL_DIR
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.joblib"))
