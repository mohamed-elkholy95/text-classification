"""Interactive classification page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import numpy as np

st.title("💬 Classify Text")

st.markdown("Enter text below to get a classification prediction.")

text_input = st.text_area("Text to classify", "This product is amazing and I love it!", height=120)

model_choice = st.selectbox("Model", ["Logistic Regression", "Naive Bayes", "SVM", "Random Forest", "Ensemble"])

if st.button("Classify", type="primary"):
    # Mock prediction for demo
    rng = np.random.default_rng(42)
    positive_words = ["love", "great", "amazing", "excellent", "wonderful", "best", "fantastic", "good"]
    is_positive = any(w in text_input.lower() for w in positive_words)
    confidence = 0.9 if is_positive else 0.85

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Class", "Positive ✅" if is_positive else "Negative ❌")
        st.metric("Confidence", f"{confidence:.1%}")
    with col2:
        st.metric("Model", model_choice)
        st.metric("Processing Time", "12ms")

    if is_positive:
        st.success("This text expresses a **positive** sentiment.")
    else:
        st.error("This text expresses a **negative** sentiment.")
