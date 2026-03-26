"""Dataset loading for text classification."""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, RAW_DIR

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_samples: int = 5000, n_classes: int = 2, seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic text classification data.

    Args:
        n_samples: Number of samples.
        n_classes: Number of classes (2 for binary).
        seed: Random seed.

    Returns:
        DataFrame with 'text' and 'label' columns.
    """
    rng = np.random.default_rng(seed)
    templates = {
        0: ["This is great and wonderful", "I love this product so much",
            "Amazing quality and service", "Highly recommend to everyone",
            "The best experience I ever had", "Fantastic work done here",
            "Very satisfied with the results", "Outstanding performance overall"],
        1: ["This is terrible and awful", "I hate this product so much",
            "Worst quality and service ever", "Do not recommend at all",
            "The worst experience I have had", "Horrible work done here",
            "Very disappointed with results", "Poor performance overall"],
        2: ["The product is okay I guess", "Nothing special about this one",
            "Average quality at best", "It works but nothing great",
            "Mixed feelings about this purchase", "Could be better",
            "Not bad not good either", "Standard average product"],
    }
    texts, labels = [], []
    for _ in range(n_samples):
        cls = rng.integers(0, min(n_classes, 3))
        base = rng.choice(templates.get(cls, templates[0]))
        all_phrases = []
        for v in templates.values():
            all_phrases.extend(v)
        noise_phrases = rng.choice(all_phrases, size=rng.integers(1, 4)).tolist()
        text = " ".join([base] + noise_phrases)
        texts.append(text)
        labels.append(cls)

    df = pd.DataFrame({"text": texts, "label": labels})
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info("Generated synthetic data: %d samples, %d classes", len(df), n_classes)
    return df


def load_sms_spam(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load SMS Spam collection dataset.

    Args:
        data_path: Path to CSV/TSV file. If None, generates synthetic.

    Returns:
        DataFrame with 'text' and 'label' columns.
    """
    path = Path(data_path) if data_path else RAW_DIR / "sms_spam.csv"
    if path.exists():
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
        df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
        logger.info("Loaded SMS Spam: %d samples", len(df))
        return df
    return generate_synthetic_data()


def load_custom(data_path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Load custom CSV dataset.

    Args:
        data_path: Path to CSV file.
        text_col: Name of text column.
        label_col: Name of label column.

    Returns:
        DataFrame with 'text' and 'label' columns.
    """
    df = pd.read_csv(data_path)
    df = df.rename(columns={text_col: "text", label_col: "label"})
    logger.info("Loaded custom dataset: %d samples from %s", len(df), data_path)
    return df


def load_dataset(name: str = "synthetic", **kwargs) -> pd.DataFrame:
    """Load a named dataset.

    Args:
        name: Dataset name ('synthetic', 'sms_spam', 'custom').
        **kwargs: Passed to specific loader.

    Returns:
        DataFrame with 'text' and 'label' columns.
    """
    loaders = {
        "synthetic": lambda: generate_synthetic_data(**kwargs),
        "sms_spam": lambda: load_sms_spam(**kwargs),
        "custom": lambda: load_custom(**kwargs),
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders.keys())}")
    return loaders[name]()


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Compute dataset statistics."""
    return {
        "n_samples": len(df),
        "n_classes": df["label"].nunique(),
        "class_distribution": df["label"].value_counts().to_dict(),
        "avg_text_length": round(float(df["text"].str.len().mean()), 1),
        "min_text_length": int(df["text"].str.len().min()),
        "max_text_length": int(df["text"].str.len().max()),
    }


def validate_dataset(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    min_samples_per_class: int = 5,
) -> dict:
    """Run data quality checks and return a diagnostic report.

    Data quality issues are one of the most common (and frustrating)
    sources of poor model performance.  This function catches problems
    **before** training so you get clear error messages instead of
    mysterious NaN losses or silently degraded accuracy.

    Checks performed:
    - Missing columns (text or label not present)
    - Null / NaN values in text or label columns
    - Empty strings in the text column
    - Duplicate rows (exact text + label matches)
    - Class imbalance (any class below *min_samples_per_class*)

    Args:
        df: Input DataFrame.
        text_col: Name of the text column.
        label_col: Name of the label column.
        min_samples_per_class: Minimum samples required per class
            before a warning is raised.  Defaults to 5.

    Returns:
        Dictionary with:
        - ``"valid"`` (bool): True if no critical issues found.
        - ``"warnings"`` (list[str]): Non-fatal issues.
        - ``"errors"`` (list[str]): Fatal issues that should block
          training.
        - ``"stats"`` (dict): Quick summary statistics.

    Example::

        >>> report = validate_dataset(df)
        >>> if not report["valid"]:
        ...     for err in report["errors"]:
        ...         print(f"ERROR: {err}")
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ── Column existence ──────────────────────────────────────────
    if text_col not in df.columns:
        errors.append(f"Missing text column '{text_col}'.")
    if label_col not in df.columns:
        errors.append(f"Missing label column '{label_col}'.")

    # If essential columns are missing, return early — other checks
    # would raise KeyError.
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings, "stats": {}}

    n_total = len(df)

    # ── Null values ───────────────────────────────────────────────
    null_text = int(df[text_col].isna().sum())
    null_label = int(df[label_col].isna().sum())
    if null_text > 0:
        errors.append(f"{null_text} null values in '{text_col}' column ({null_text / n_total:.1%}).")
    if null_label > 0:
        errors.append(f"{null_label} null values in '{label_col}' column ({null_label / n_total:.1%}).")

    # ── Empty strings ─────────────────────────────────────────────
    empty_text = int((df[text_col].astype(str).str.strip() == "").sum())
    if empty_text > 0:
        warnings.append(f"{empty_text} empty text entries ({empty_text / n_total:.1%}).")

    # ── Duplicates ────────────────────────────────────────────────
    n_dupes = int(df.duplicated(subset=[text_col, label_col]).sum())
    if n_dupes > 0:
        warnings.append(f"{n_dupes} duplicate (text, label) pairs ({n_dupes / n_total:.1%}).")

    # ── Class balance ─────────────────────────────────────────────
    class_counts = df[label_col].value_counts()
    small_classes = class_counts[class_counts < min_samples_per_class]
    if len(small_classes) > 0:
        for cls, count in small_classes.items():
            warnings.append(
                f"Class '{cls}' has only {count} sample(s) "
                f"(min recommended: {min_samples_per_class})."
            )

    stats = {
        "n_samples": n_total,
        "n_classes": int(df[label_col].nunique()),
        "null_text": null_text,
        "null_label": null_label,
        "empty_text": empty_text,
        "duplicates": n_dupes,
        "class_distribution": class_counts.to_dict(),
    }

    is_valid = len(errors) == 0
    if not is_valid:
        logger.error("Dataset validation FAILED: %s", "; ".join(errors))
    elif warnings:
        logger.warning("Dataset validation passed with warnings: %s", "; ".join(warnings))
    else:
        logger.info("Dataset validation passed (%d samples, %d classes).", n_total, stats["n_classes"])

    return {"valid": is_valid, "errors": errors, "warnings": warnings, "stats": stats}
