"""Overview page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px

st.title("📊 Text Classification — Overview")
st.markdown("""
**Multi-model text classification pipeline** with TF-IDF features, four baseline models,
transformer support, hyperparameter tuning, and interactive exploration.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Features")
    st.markdown("- TF-IDF with configurable n-grams\n- Sentence embeddings (with fallback)\n- Feature combination & ensemble voting")
    st.subheader("Models")
    st.markdown("- Logistic Regression\n- Naive Bayes\n- Linear SVM\n- Random Forest\n- Transformer (DistilBERT)")
with col2:
    st.subheader("Tools")
    st.markdown("- Optuna hyperparameter tuning\n- Soft/Hard voting ensemble\n- Text augmentation (NLTK)\n- REST API (FastAPI)")
    st.subheader("Metrics")
    st.markdown("- Accuracy, Precision, Recall, F1\n- ROC-AUC, PR-AUC\n- Confusion Matrix\n- Classification Report")

# Demo chart
import numpy as np
models = ["LogReg", "Naive Bayes", "SVM", "RF", "Ensemble"]
scores = [0.89, 0.87, 0.91, 0.85, 0.92]
fig = px.bar(x=models, y=scores, labels={"x": "Model", "y": "F1 Score"},
             title="Model Performance Comparison", color=scores, color_continuous_scale="Blues")
fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#262730", font_color="white")
st.plotly_chart(fig, use_container_width=True)
