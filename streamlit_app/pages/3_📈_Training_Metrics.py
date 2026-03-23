"""📈 Training Metrics — Real model evaluation with visualizations.

This page demonstrates how to evaluate trained ML models:
  1. Train models on synthetic data (cached with @st.cache_resource)
  2. Compute per-class metrics (precision, recall, F1 per class)
  3. Display a confusion matrix heatmap (Plotly)
  4. Compare all models on accuracy, precision, recall, F1 (grouped bar chart)

Educational notes:
  - Per-class metrics reveal if a model is biased toward majority classes.
  - The confusion matrix shows WHERE the model makes mistakes.
  - Comparing models helps you choose the right tool for your problem.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path for src module imports.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data.dataset_loader import generate_synthetic_data
from src.data.preprocessor import TextPreprocessor
from src.features.tfidf_features import TfidfFeatureExtractor
from src.models.baseline_models import (
    train_naive_bayes,
    train_logistic_regression,
    train_random_forest,
)

st.title("📈 Training Metrics")
st.markdown(
    "Real evaluation metrics from models trained on synthetic data. "
    "Explore per-class performance, confusion matrices, and model comparisons."
)


# ---------------------------------------------------------------------------
# Cached training pipeline
#
# @st.cache_resource ensures we train only once per app session.
# The hash is based on the function source code + arguments, so editing
# the code or changing n_samples/seed will invalidate the cache.
# ---------------------------------------------------------------------------

@st.cache_resource
def train_and_evaluate():
    """Train all models on synthetic data and compute evaluation metrics.

    Returns:
        Tuple of (class_names, results_list, confusion_matrices).
        - class_names: list of decoded class label strings
        - results_list: list of dicts with model_name, overall metrics, and per-class df
        - confusion_matrices: dict of model_name → 2D numpy array
    """
    # --- Step 1: Load & preprocess data ---
    df = generate_synthetic_data(n_samples=3000, n_classes=3, seed=42)
    df["label"] = df["label"].astype(str)

    preprocessor = TextPreprocessor()
    df_clean, y = preprocessor.fit_transform(df)
    class_names = preprocessor.classes.tolist()

    # --- Step 2: TF-IDF features ---
    tfidf = TfidfFeatureExtractor()
    X = tfidf.fit_transform(df_clean["text"].tolist())

    # --- Step 3: Train models and evaluate ---
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    models_to_train = {
        "Naive Bayes": lambda: train_naive_bayes(X, y, alpha=1.0),
        "Logistic Regression": lambda: train_logistic_regression(X, y, C=1.0),
        "Random Forest": lambda: train_random_forest(X, y, n_estimators=200, max_depth=10),
    }

    results = []
    cm_dict = {}

    for model_name, train_fn in models_to_train.items():
        model = train_fn()
        y_pred = model.predict(X)

        # Overall metrics (weighted average for multi-class)
        overall = {
            "Model": model_name,
            "Accuracy": round(float(accuracy_score(y, y_pred)), 4),
            "Precision": round(float(precision_score(y, y_pred, average="weighted", zero_division=0)), 4),
            "Recall": round(float(recall_score(y, y_pred, average="weighted", zero_division=0)), 4),
            "F1 Score": round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 4),
        }

        # Per-class metrics (for the detailed table)
        report = classification_report(y, y_pred, target_names=class_names, zero_division=0, output_dict=True)
        per_class_rows = []
        for cls in class_names:
            if cls in report:
                per_class_rows.append({
                    "Class": cls,
                    "Precision": round(report[cls]["precision"], 4),
                    "Recall": round(report[cls]["recall"], 4),
                    "F1 Score": round(report[cls]["f1-score"], 4),
                    "Support": report[cls]["support"],
                })
        per_class_df = pd.DataFrame(per_class_rows)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)

        results.append({
            "model_name": model_name,
            "overall": overall,
            "per_class_df": per_class_df,
        })
        cm_dict[model_name] = cm

    return class_names, results, cm_dict


# --- Load cached results ---
class_names, results, cm_dict = train_and_evaluate()


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
selected_model = st.selectbox(
    "Select a model to inspect",
    [r["model_name"] for r in results],
    index=0,
    help="Choose which model's metrics to view in detail.",
)

# Find the selected result
result = next(r for r in results if r["model_name"] == selected_model)


# ---------------------------------------------------------------------------
# Section 1: Overall Metrics Summary
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader(f"📊 Overall Metrics — {selected_model}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", f"{result['overall']['Accuracy']:.2%}")
with col2:
    st.metric("Precision", f"{result['overall']['Precision']:.2%}")
with col3:
    st.metric("Recall", f"{result['overall']['Recall']:.2%}")
with col4:
    st.metric("F1 Score", f"{result['overall']['F1 Score']:.2%}")

st.caption(
    "Educational note: **Accuracy** = correct predictions / total. "
    "**Precision** = true positives / (true positives + false positives). "
    "**Recall** = true positives / (true positives + false negatives). "
    "**F1** = harmonic mean of precision & recall."
)


# ---------------------------------------------------------------------------
# Section 2: Per-Class Metrics Table
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📋 Per-Class Metrics")

st.markdown(
    "Per-class metrics show how the model performs on **each class individually**. "
    "This is critical for imbalanced datasets where overall accuracy can be misleading."
)

st.dataframe(
    result["per_class_df"],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Precision": st.column_config.ProgressColumn("Precision", format="%.2f", min_value=0, max_value=1),
        "Recall": st.column_config.ProgressColumn("Recall", format="%.2f", min_value=0, max_value=1),
        "F1 Score": st.column_config.ProgressColumn("F1 Score", format="%.2f", min_value=0, max_value=1),
    },
)


# ---------------------------------------------------------------------------
# Section 3: Confusion Matrix (Plotly Heatmap)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Confusion Matrix")

st.markdown(
    "The **confusion matrix** shows predicted vs. actual labels. "
    "The diagonal (top-left → bottom-right) contains correct predictions. "
    "Off-diagonal cells are **misclassifications** — the darker they are, "
    "the more frequent the mistake."
)

cm = cm_dict[selected_model]

fig_cm = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16, "color": "white"},
        hoverongaps=False,
    )
)
fig_cm.update_layout(
    title=f"Confusion Matrix — {selected_model}",
    xaxis_title="Predicted Label",
    yaxis_title="True Label",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#262730",
    font_color="white",
    height=450,
    width=500,
)
st.plotly_chart(fig_cm, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 4: Model Comparison (All Models)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("⚖️ Model Comparison")

st.markdown(
    "Comparing models side by side reveals which algorithm performs best "
    "for this specific dataset. In practice, you'd also compare training time "
    "and inference latency."
)

# Build a DataFrame of all models' overall metrics
comparison_df = pd.DataFrame([r["overall"] for r in results])

# Display as a table
st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Model": st.column_config.TextColumn("Model"),
        "Accuracy": st.column_config.ProgressColumn("Accuracy", format="%.2f", min_value=0, max_value=1),
        "Precision": st.column_config.ProgressColumn("Precision", format="%.2f", min_value=0, max_value=1),
        "Recall": st.column_config.ProgressColumn("Recall", format="%.2f", min_value=0, max_value=1),
        "F1 Score": st.column_config.ProgressColumn("F1 Score", format="%.2f", min_value=0, max_value=1),
    },
)

# Grouped bar chart: all models × all metrics
metrics_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig_compare = go.Figure()
for metric, color in zip(metrics_cols, colors):
    fig_compare.add_trace(
        go.Bar(
            x=comparison_df["Model"],
            y=comparison_df[metric],
            name=metric,
            marker_color=color,
        )
    )

fig_compare.update_layout(
    title="Model Comparison — All Metrics",
    barmode="group",
    xaxis_title="Model",
    yaxis_title="Score",
    yaxis=dict(range=[0, 1.05]),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    paper_bgcolor="#0e1117",
    plot_bgcolor="#262730",
    font_color="white",
)
st.plotly_chart(fig_compare, use_container_width=True)
