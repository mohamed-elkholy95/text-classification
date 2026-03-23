"""💬 Classify Text — Real-time multi-model predictions.

This page demonstrates the full ML pipeline:
  1. Load synthetic data → preprocess → extract TF-IDF features
  2. Train baseline models (Naive Bayes, Logistic Regression, Random Forest)
  3. Accept user text input
  4. Show side-by-side predictions with confidence bars
  5. Display top TF-IDF features that contributed to the classification

Educational note: @st.cache_resource ensures models are trained only ONCE
and reused across reruns — similar to model serving in production.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so we can import `src.*` modules
# from anywhere the Streamlit server is launched.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import numpy as np
import pandas as pd

from src.data.dataset_loader import generate_synthetic_data
from src.data.preprocessor import TextPreprocessor
from src.features.tfidf_features import TfidfFeatureExtractor
from src.models.baseline_models import (
    train_naive_bayes,
    train_logistic_regression,
    train_random_forest,
)

st.title("💬 Classify Text")
st.markdown(
    "Enter any text below and watch **three different models** predict its class "
    "side by side. This is a core ML concept — comparing model agreement and "
    "confidence helps you understand which model to trust."
)


# ---------------------------------------------------------------------------
# Cached pipeline: data loading → preprocessing → TF-IDF → model training
#
# Why @st.cache_resource?
#   - Trains models exactly ONCE and caches the result in memory.
#   - Subsequent reruns (e.g., when the user types new text) reuse the
#     trained models without retraining — just like a production API.
#   - Streamlit clears the cache when the script hash changes (code edit).
# ---------------------------------------------------------------------------

@st.cache_resource
def build_pipeline():
    """Build and cache the full training pipeline.

    Returns:
        Tuple of (preprocessor, tfidf, models_dict, vectorizer).
    """
    # Step 1 — Generate synthetic dataset (positive / negative / neutral)
    # In a real project, you'd load from CSV via load_dataset("sms_spam").
    df = generate_synthetic_data(n_samples=3000, n_classes=3, seed=42)
    df["label"] = df["label"].astype(str)  # preprocessor expects strings

    # Step 2 — Clean text & encode labels
    preprocessor = TextPreprocessor()
    df_clean, y = preprocessor.fit_transform(df)

    # Step 3 — Extract TF-IDF features (unigrams + bigrams, sublinear TF)
    tfidf = TfidfFeatureExtractor()
    X = tfidf.fit_transform(df_clean["text"].tolist())

    # Step 4 — Train each baseline model on the same feature matrix
    models = {
        "Naive Bayes": train_naive_bayes(X, y, alpha=1.0),
        "Logistic Regression": train_logistic_regression(X, y, C=1.0),
        "Random Forest": train_random_forest(X, y, n_estimators=200, max_depth=10),
    }

    return preprocessor, tfidf, models


@st.cache_resource
def get_preprocessor():
    """Return just the preprocessor (for class label decoding)."""
    return build_pipeline()[0]


# Build the pipeline (cached after first call)
preprocessor, tfidf_extractor, trained_models = build_pipeline()
class_names = preprocessor.classes.tolist()


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------
st.subheader("📝 Enter Your Text")

# Some example prompts to help the user get started
example_texts = {
    "Positive review": "This product is amazing and I love it so much!",
    "Negative review": "Terrible quality, worst experience ever, very disappointed.",
    "Neutral review": "The product is okay, nothing special but it works fine.",
}

with st.expander("💡 Try an example", expanded=False):
    for label, txt in example_texts.items():
        if st.button(label, key=f"example_{label}"):
            st.session_state["user_text"] = txt

default_text = st.session_state.get("user_text", "This product is amazing!")
user_text = st.text_area(
    "Text to classify",
    value=default_text,
    height=120,
    key="classify_input",
    help="Type or paste any text. The models will predict its sentiment class.",
)

# ---------------------------------------------------------------------------
# Run predictions when the user clicks Classify
# ---------------------------------------------------------------------------
if st.button("🔍 Classify", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        # --- Preprocess the user's input the same way as training data ---
        cleaned = preprocessor.clean_text(user_text)
        X_input = tfidf_extractor.transform([cleaned])

        # --- Collect predictions from every model ---
        predictions = []
        for model_name, model in trained_models.items():
            pred_class = model.predict(X_input)[0]
            decoded_label = preprocessor.decode_labels(np.array([pred_class]))[0]

            # Confidence: max probability (for models that support predict_proba)
            # Naive Bayes & Logistic Regression have predict_proba; Random Forest too.
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_input)[0]
                confidence = float(np.max(proba))
                class_probs = dict(zip(class_names, proba.tolist()))
            else:
                confidence = 1.0  # SVM uses decision_function, not proba
                class_probs = {decoded_label: 1.0}

            predictions.append({
                "model": model_name,
                "prediction": decoded_label,
                "confidence": confidence,
                "class_probs": class_probs,
            })

        # --- Show all predictions side by side ---
        st.markdown("---")
        st.subheader("📊 Model Predictions")

        # Educational note: displaying all models side-by-side shows that
        # different algorithms can disagree — a key insight in ensemble methods.
        cols = st.columns(len(predictions))
        for i, pred in enumerate(predictions):
            with cols[i]:
                # Pick an emoji based on the predicted class
                emoji_map = {"0": "😊", "1": "😠", "2": "😐"}
                emoji = emoji_map.get(str(pred["prediction"]), "🏷️")

                st.markdown(
                    f"**{pred['model']}**\n\n"
                    f"### {emoji} {pred['prediction']}"
                )
                # Confidence bar using st.progress (0.0 → 1.0)
                st.progress(pred["confidence"])
                st.caption(f"Confidence: **{pred['confidence']:.1%}**")

                # Show per-class probability breakdown
                if len(pred["class_probs"]) > 1:
                    for cls, prob in sorted(
                        pred["class_probs"].items(), key=lambda x: x[1], reverse=True
                    ):
                        st.markdown(f"- Class **{cls}**: {prob:.1%}")

        # --- Top TF-IDF features for the input ---
        st.markdown("---")
        st.subheader("🔤 Top TF-IDF Features")

        # The TF-IDF vectorizer stores the learned vocabulary.
        # We can look up which features are most active for this input.
        feature_names = tfidf_extractor.get_feature_names()
        dense_input = X_input.toarray().flatten()

        # Get indices of top 5 features by TF-IDF score
        top_indices = np.argsort(dense_input)[-5:][::-1]
        top_features = [
            (feature_names[idx], dense_input[idx])
            for idx in top_indices
            if dense_input[idx] > 0
        ]

        if top_features:
            st.markdown(
                "These are the **most important words/bigrams** (by TF-IDF score) "
                "in your input text:"
            )
            for rank, (feat, score) in enumerate(top_features, 1):
                st.markdown(f"**{rank}.** `{feat}` — score: **{score:.4f}**")

            # Visual bar chart of top features
            fig_features = st.session_state.get("_last_feat_fig")  # reuse if exists

            import plotly.graph_objects as go

            feat_names_plot = [f[0] for f in top_features]
            feat_scores_plot = [f[1] for f in top_features]

            fig = go.Figure(
                go.Bar(
                    x=feat_scores_plot,
                    y=feat_names_plot,
                    orientation="h",
                    marker_color="#1f77b4",
                )
            )
            fig.update_layout(
                title="Top 5 TF-IDF Features for Your Input",
                xaxis_title="TF-IDF Score",
                yaxis_title="Feature",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#262730",
                font_color="white",
                height=300,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No known TF-IDF features found in the input (all scores are 0).")

        # --- Consensus check ---
        all_preds = [p["prediction"] for p in predictions]
        if len(set(all_preds)) == 1:
            st.success(
                f"✅ **All models agree!** Predicted class: **{all_preds[0]}**"
            )
        else:
            st.warning(
                f"⚠️ **Models disagree!** Predictions: "
                + ", ".join(f"**{p['model']}** → {p['prediction']}" for p in predictions)
            )
            st.caption(
                "Model disagreement is normal and educational — it's why ensemble "
                "methods (voting, averaging) often outperform any single model."
            )
