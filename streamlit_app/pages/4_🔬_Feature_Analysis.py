"""Feature analysis page for the text classification dashboard.

This page provides an interactive exploration of TF-IDF feature importance
derived from a trained Logistic Regression model, along with corpus-level
statistics such as text length distributions and word frequency charts.

All heavy computations (data generation, model training, feature extraction)
are cached with ``@st.cache_resource`` so they run only once per session.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports resolve correctly
# when Streamlit executes this page in isolation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.dataset_loader import generate_synthetic_data
from src.data.preprocessor import TextPreprocessor
from src.features.tfidf_features import TfidfFeatureExtractor
from src.models.baseline_models import train_logistic_regression

# ── Dark theme palette (matches STREAMLIT_THEME in src/config.py) ──────────
_BG_PAPER = "#0e1117"       # Streamlit background
_BG_PLOT = "#262730"         # Secondary background
_CLR_TEXT = "#ffffff"
_CLR_ACCENT = "#1f77b4"      # Streamlit default blue
_CLR_GRID = "#444444"

# ── Cached data / model loading ────────────────────────────────────────────

@st.cache_resource(show_spinner="Generating synthetic dataset…")
def load_data() -> pd.DataFrame:
    """Generate and cache the synthetic text classification dataset.

    Returns:
        DataFrame with 'text' and 'label' columns.
    """
    return generate_synthetic_data(n_samples=5000, n_classes=3)


@st.cache_resource(show_spinner="Training feature extractor and model…")
def build_features_and_model(
    texts: List[str],
    labels: np.ndarray,
) -> Tuple[TfidfFeatureExtractor, object, List[str]]:
    """Build TF-IDF features, train a Logistic Regression, and return results.

    The trained model's ``coef_`` attribute gives us the real feature
    importance — the weight each TF-IDF term has in the decision boundary.

    Args:
        texts: Cleaned text strings.
        labels: Integer-encoded labels.

    Returns:
        Tuple of (fitted_extractor, fitted_model, feature_names).
    """
    extractor = TfidfFeatureExtractor()
    X = extractor.fit_transform(texts)
    model = train_logistic_regression(X, labels)
    feature_names = extractor.get_feature_names()
    return extractor, model, feature_names


# ── Page layout ────────────────────────────────────────────────────────────

st.title("🔬 Feature Analysis")
st.markdown(
    "Explore **real TF-IDF feature importance** from a trained Logistic "
    "Regression model, plus corpus-level text statistics from the synthetic "
    "dataset."
)

# Load data once (cached)
df_raw = load_data()

# Preprocess: clean text and encode labels
preprocessor = TextPreprocessor()
df_clean, labels = preprocessor.fit_transform(df_raw)

texts = df_clean["text"].tolist()
class_names = list(map(str, preprocessor.classes)) if len(preprocessor.classes) > 0 else ["0", "1", "2"]

# Build features & train model once (cached)
extractor, model, feature_names = build_features_and_model(texts, labels)

# ═══════════════════════════════════════════════════════════════════════════
# Section 1: TF-IDF Feature Importance (real coef_ values)
# ═══════════════════════════════════════════════════════════════════════════

st.header("TF-IDF Feature Importance (Logistic Regression Coefficients)")
st.markdown(
    "Each bar shows the absolute weight a TF-IDF term received in the "
    "trained Logistic Regression. Larger magnitude → stronger influence "
    "on the classification decision."
)

# For multiclass, coef_ has shape (n_classes, n_features).
# We take the mean absolute coefficient across all classes as a
# single importance score per feature.
coef_matrix = model.coef_  # (n_classes, n_features)
mean_abs_importance = np.mean(np.abs(coef_matrix), axis=0)

# Identify the sign associated with the most influential class for each feature
# to colour-code positive (push toward class 0) vs negative contributions.
top_n = min(20, len(feature_names))

# Sort indices by mean absolute importance (descending)
sorted_indices = np.argsort(mean_abs_importance)[::-1][:top_n]

importance_df = pd.DataFrame({
    "feature": [feature_names[i] for i in sorted_indices],
    "importance": mean_abs_importance[sorted_indices],
})

# Build a per-class contribution for tooltip enrichment (optional color)
class_contributions = {}
for cls_idx in range(coef_matrix.shape[0]):
    class_contributions[class_names[cls_idx]] = coef_matrix[cls_idx][sorted_indices]

fig_importance = px.bar(
    importance_df,
    x="importance",
    y="feature",
    orientation="h",
    title=f"Top {top_n} TF-IDF Features by Mean |Coefficient|",
    color="importance",
    color_continuous_scale="Blues_r",
    labels={"importance": "Mean |Coefficient|", "feature": "TF-IDF Term"},
)
fig_importance.update_layout(
    paper_bgcolor=_BG_PAPER,
    plot_bgcolor=_BG_PLOT,
    font_color=_CLR_TEXT,
    height=500,
    yaxis=dict(autorange="reversed"),  # Highest importance at top
    xaxis=dict(gridcolor=_CLR_GRID),
)
fig_importance.update_traces(marker_line_color=_CLR_TEXT, marker_line_width=0.5)
st.plotly_chart(fig_importance, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Per-class top features
# ═══════════════════════════════════════════════════════════════════════════

st.header("Top Features per Class")
st.markdown(
    "For each class, these are the TF-IDF terms with the **most positive**
    coefficient — i.e., the words that push the model toward predicting
    that class."
)

cols = st.columns(min(3, coef_matrix.shape[0]))
per_class_top = 8

for cls_idx, col in enumerate(cols):
    with col:
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        # Get indices of highest positive coefficients for this class
        cls_coefs = coef_matrix[cls_idx]
        top_cls_indices = np.argsort(cls_coefs)[::-1][:per_class_top]
        cls_df = pd.DataFrame({
            "feature": [feature_names[i] for i in top_cls_indices],
            "weight": cls_coefs[top_cls_indices],
        })
        fig_cls = px.bar(
            cls_df,
            x="weight",
            y="feature",
            orientation="h",
            title=f"Class: {cls_name}",
            color="weight",
            color_continuous_scale="Viridis",
        )
        fig_cls.update_layout(
            paper_bgcolor=_BG_PAPER,
            plot_bgcolor=_BG_PLOT,
            font_color=_CLR_TEXT,
            height=300,
            margin=dict(l=80),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(gridcolor=_CLR_GRID),
        )
        st.plotly_chart(fig_cls, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Section 3: Text Length Distribution
# ═══════════════════════════════════════════════════════════════════════════

st.header("Text Length Distribution")
st.markdown(
    "Histogram of character lengths from the synthetic dataset, "
    "colour-coded by class label."
)

# Compute character lengths for all documents
df_clean_copy = df_clean.copy()
df_clean_copy["char_length"] = df_clean_copy["text"].str.len()
df_clean_copy["label_str"] = df_clean_copy["label"].map(
    {i: class_names[i] for i in range(len(class_names))}
)

fig_length = px.histogram(
    df_clean_copy,
    x="char_length",
    color="label_str",
    nbins=40,
    title="Character Length Distribution by Class",
    labels={"char_length": "Characters", "label_str": "Class"},
    opacity=0.75,
    barmode="overlay",
)
fig_length.update_layout(
    paper_bgcolor=_BG_PAPER,
    plot_bgcolor=_BG_PLOT,
    font_color=_CLR_TEXT,
    xaxis=dict(gridcolor=_CLR_GRID),
    yaxis=dict(gridcolor=_CLR_GRID),
)
st.plotly_chart(fig_length, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Word Frequency Bar Chart
# ═══════════════════════════════════════════════════════════════════════════

st.header("Word Frequency (Top 25)")
st.markdown(
    "The most common words across the entire corpus after preprocessing. "
    "High-frequency stopwords may need to be filtered; distinctive words "
    "should appear in the TF-IDF importance chart above."
)

# Count word frequencies across all cleaned texts
all_words: Counter = Counter()
for text in texts:
    all_words.update(text.lower().split())

top_words = all_words.most_common(25)
word_freq_df = pd.DataFrame(top_words, columns=["word", "count"])

fig_words = px.bar(
    word_freq_df,
    x="word",
    y="count",
    title="Top 25 Most Frequent Words",
    color="count",
    color_continuous_scale="Reds_r",
    labels={"word": "Word", "count": "Frequency"},
)
fig_words.update_layout(
    paper_bgcolor=_BG_PAPER,
    plot_bgcolor=_BG_PLOT,
    font_color=_CLR_TEXT,
    xaxis=dict(gridcolor=_CLR_GRID, tickangle=45),
    yaxis=dict(gridcolor=_CLR_GRID),
)
fig_words.update_traces(marker_line_color=_CLR_TEXT, marker_line_width=0.5)
st.plotly_chart(fig_words, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Vocabulary Statistics
# ═══════════════════════════════════════════════════════════════════════════

st.header("Vocabulary & Corpus Statistics")

# Total unique words (types)
n_types = len(all_words)
# Total word tokens
n_tokens = sum(all_words.values())
# Average words per document
avg_words_per_doc = np.mean([len(t.split()) for t in texts])
# Median words per document
median_words_per_doc = np.median([len(t.split()) for t in texts])

# Unique words that appear only once (hapax legomena)
n_hapax = sum(1 for count in all_words.values() if count == 1)

# Lexical diversity = types / tokens (higher → more diverse vocabulary)
lexical_diversity = n_types / n_tokens if n_tokens > 0 else 0

# Average document length in characters
avg_char_len = float(df_clean_copy["char_length"].mean())

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Vocabulary (Types)", f"{n_types:,}")
    st.metric("Total Word Tokens", f"{n_tokens:,}")
    st.metric("TF-IDF Features Extracted", f"{len(feature_names):,}")

with col2:
    st.metric("Avg Words / Document", f"{avg_words_per_doc:.1f}")
    st.metric("Median Words / Document", f"{median_words_per_doc:.1f}")
    st.metric("Hapax Legomena (freq=1)", f"{n_hapax:,}")

st.metric("Lexical Diversity (types/tokens)", f"{lexical_diversity:.4f}")
st.metric("Avg Document Length (chars)", f"{avg_char_len:.1f}")

# Per-class word count summary
st.subheader("Per-Class Word Statistics")
class_stats: List[Dict[str, object]] = []
for cls_label in sorted(df_clean_copy["label"].unique()):
    cls_texts = df_clean_copy[df_clean_copy["label"] == cls_label]["text"]
    cls_word_counts = [len(t.split()) for t in cls_texts]
    cls_name = class_names[cls_label] if cls_label < len(class_names) else str(cls_label)
    class_stats.append({
        "Class": cls_name,
        "Documents": len(cls_texts),
        "Avg Words": round(np.mean(cls_word_counts), 1),
        "Median Words": round(np.median(cls_word_counts), 1),
        "Total Tokens": sum(cls_word_counts),
    })

st.dataframe(
    pd.DataFrame(class_stats),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "💡 **Tip:** Vocabulary statistics help you spot data-quality issues. "
    "A very high hapax count relative to vocabulary size suggests many "
    "typos or rare terms that TF-IDF may not help with — consider "
    "lowering `min_df` or adding a stopword list."
)
