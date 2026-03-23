"""📊 Overview Page — Educational introduction to text classification.

This page serves as the landing page for the Streamlit portfolio app.
It covers:
  - What is Text Classification?
  - How TF-IDF works (the feature extraction method used in this project)
  - Why we train and compare multiple models (bias-variance tradeoff)
  - What the user will learn by exploring this project

Design choices:
  - Uses structured markdown boxes and visual layout for readability.
  - Dark-theme compatible colors throughout.
  - Architecture diagram rendered as a structured text diagram.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so we can import src modules if needed.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.title("📊 Text Classification — Overview")
st.markdown(
    "A **multi-model text classification pipeline** with TF-IDF features, "
    "four baseline models, transformer support, hyperparameter tuning, and "
    "interactive exploration."
)

# ---------------------------------------------------------------------------
# Architecture Diagram
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🏗️ Pipeline Architecture")

# Render a text-based architecture diagram inside a code block for monospace.
st.code(
    """
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT CLASSIFICATION PIPELINE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐      │
│  │  Raw     │───▶│ Preprocessor │───▶│ TF-IDF Feature    │      │
│  │  Text    │    │ (clean,      │    │ Extraction        │      │
│  │  Data    │    │  tokenize)   │    │ (unigrams+bigrams)│      │
│  └──────────┘    └──────────────┘    └────────┬──────────┘      │
│                                                │                 │
│                      ┌─────────────────────────┼───────────┐    │
│                      │                         │           │    │
│                 ┌────▼─────┐  ┌──────────┐ ┌──▼──────┐ ┌──▼──┐  │
│                 │  Naive   │  │ Logistic │ │ Linear  │ │ RF  │  │
│                 │  Bayes   │  │ Regress. │ │ SVM     │ │     │  │
│                 └────┬─────┘  └────┬─────┘ └────┬────┘ └──┬──┘  │
│                      │             │            │         │     │
│                      └─────────────┼────────────┼─────────┘    │
│                                    │            │               │
│                              ┌─────▼────────────▼─────┐         │
│                              │    Ensemble / Voting   │         │
│                              │    (Soft or Hard)      │         │
│                              └───────────┬────────────┘         │
│                                          │                      │
│                                   ┌──────▼──────┐               │
│                                   │  Prediction │               │
│                                   │  + Metrics  │               │
│                                   └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
    """,
    language=None,  # plain text, no syntax highlighting
)

# ---------------------------------------------------------------------------
# "What You'll Learn" — Key ML concepts covered in this project
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🎓 What You'll Learn")

learn_items = [
    ("📝 Text Preprocessing", "Cleaning, tokenization, and normalization — why raw text can't be fed directly into ML models."),
    ("🔢 TF-IDF Features", "Converting text into numerical vectors. Term Frequency × Inverse Document Frequency measures word importance."),
    ("🤖 Multiple Classifiers", "Training and comparing Naive Bayes, Logistic Regression, SVM, and Random Forest on the same data."),
    ("📊 Evaluation Metrics", "Accuracy, Precision, Recall, F1-Score, ROC-AUC — and why accuracy alone is misleading for imbalanced data."),
    ("🎯 Hyperparameter Tuning", "Using Optuna to find optimal model settings automatically."),
    ("🧠 Ensembles", "Combining multiple models via soft/hard voting for better predictions."),
    ("⚡ REST API", "Serving predictions via FastAPI — the bridge between ML research and production."),
    ("📈 Feature Analysis", "Understanding *why* models make predictions by inspecting TF-IDF weights and feature importance."),
]

cols = st.columns(2)
for i, (title, desc) in enumerate(learn_items):
    with cols[i % 2]:
        st.markdown(f"**{title}**\n\n{desc}")

# ---------------------------------------------------------------------------
# Educational Concepts — Expandable sections
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📚 Key Concepts")

# --- Concept 1: What is Text Classification? ---
with st.expander("🏷️ What is Text Classification?", expanded=False):
    st.markdown(
        """
**Text classification** is the task of assigning a category (or label) to a
piece of text. Common examples include:

| Application | Classes |
|---|---|
| Spam detection | Spam / Not Spam |
| Sentiment analysis | Positive / Negative / Neutral |
| Topic labeling | Sports / Tech / Politics / ... |
| Intent recognition | Book Flight / Cancel Order / ... |

**How it works:**
1. **Represent** text as numbers (TF-IDF, embeddings, etc.)
2. **Train** a model on labeled examples
3. **Predict** the class of new, unseen text

The fundamental challenge is that ML models can't read — they need
mathematical representations of language.
        """
    )

# --- Concept 2: TF-IDF Explained ---
with st.expander("🔤 TF-IDF Explained", expanded=False):
    st.markdown(
        """
**TF-IDF** stands for **Term Frequency – Inverse Document Frequency**.
It converts text into a numerical vector where each dimension represents
a word (or n-gram).

**Term Frequency (TF):** How often a word appears in THIS document.
```
TF("great", doc) = count of "great" in doc / total words in doc
```

**Inverse Document Frequency (IDF):** How RARE a word is across ALL documents.
Words like "the", "is", "a" appear everywhere → low IDF.
Words like "fantastic", "terrible" are selective → high IDF.
```
IDF("great") = log(total docs / docs containing "great")
```

**TF-IDF = TF × IDF**

This gives high scores to words that are:
- **Frequent in the current document** (TF is high)
- **Rare across the corpus** (IDF is high)

In this project, we also use **bigrams** (pairs of consecutive words like
"very good", "not great") to capture context that single words miss.

**Example:** In a review dataset, the word *"terrible"* has high TF-IDF for
negative reviews because it's rare overall but frequent in negatives.
        """
    )

# --- Concept 3: Why Multiple Models? ---
with st.expander("🤔 Why Multiple Models?", expanded=False):
    st.markdown(
        """
Different models learn differently — that's the whole point of comparing them.

| Model | Strength | Weakness |
|---|---|---|
| **Naive Bayes** | Fast, works well with small data | Assumes feature independence (rarely true) |
| **Logistic Regression** | Interpretable, calibrated probabilities | Can underfit complex patterns |
| **Linear SVM** | Good margin maximization | No native probability estimates |
| **Random Forest** | Captures non-linear interactions | Slower, less interpretable |
| **Ensemble** | Often best overall | More complex to deploy |

**Key insight:** No single model is best for every dataset. Comparing them
teaches you about the **bias-variance tradeoff**:

- **High bias** (Naive Bayes, LogReg) → may underfit but generalize well
- **High variance** (Random Forest) → can overfit but capture complexity

By training multiple models on the same data, you learn *which assumptions*
work for your specific problem.
        """
    )

# ---------------------------------------------------------------------------
# Project Quick Stats (real data)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Quick Stats")

from src.data.dataset_loader import generate_synthetic_data, get_dataset_stats

# Generate a small dataset just for stats display (fast, cached by Streamlit)
with st.spinner("Loading dataset statistics..."):
    df = generate_synthetic_data(n_samples=3000, n_classes=3, seed=42)
    stats = get_dataset_stats(df)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Samples", f"{stats['n_samples']:,}")
with col2:
    st.metric("Classes", stats["n_classes"])
with col3:
    st.metric("Avg Text Length", f"{stats['avg_text_length']} chars")
with col4:
    st.metric("Vocabulary", "Varies by TF-IDF config")

# Class distribution
st.markdown("**Class Distribution:**")
dist = stats["class_distribution"]
class_labels = {0: "😊 Positive", 1: "😠 Negative", 2: "😐 Neutral"}
for cls_id, count in sorted(dist.items()):
    label = class_labels.get(cls_id, f"Class {cls_id}")
    pct = count / stats["n_samples"] * 100
    st.markdown(f"- **{label}**: {count:,} samples ({pct:.1f}%)")

# ---------------------------------------------------------------------------
# Navigation hints
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "👉 **Next steps:** Use the sidebar to navigate to **💬 Classify** (try it live!), "
    "**📈 Training Metrics** (see model performance), or **🔬 Feature Analysis** "
    "(understand what the models learned)."
)
