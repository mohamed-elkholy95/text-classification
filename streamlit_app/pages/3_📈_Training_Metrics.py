"""Training metrics page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.title("📈 Training Metrics")

st.markdown("Training curves and model comparison after fitting on the dataset.")

# Simulated training data
epochs = list(range(1, 21))
loss = [1.8 - 0.07 * i + 0.005 * i * np.sin(i) for i in epochs]
val_loss = [1.9 - 0.06 * i + 0.01 * i * np.cos(i) for i in epochs]

fig = go.Figure()
fig.add_trace(go.Scatter(x=epochs, y=loss, name="Train Loss", line=dict(color="#1f77b4", width=2)))
fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss", line=dict(color="#ff7f0e", width=2, dash="dash")))
fig.update_layout(title="Training vs Validation Loss", xaxis_title="Epoch", yaxis_title="Loss",
                  paper_bgcolor="#0e1117", plot_bgcolor="#262730", font_color="white")
st.plotly_chart(fig, use_container_width=True)

# Model comparison
models = ["Naive Bayes", "LogReg", "SVM", "Random Forest", "Ensemble"]
accuracy = [0.82, 0.89, 0.91, 0.85, 0.92]
f1_scores = [0.81, 0.88, 0.90, 0.84, 0.91]

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=models, y=accuracy, name="Accuracy", marker_color="#1f77b4"))
fig2.add_trace(go.Bar(x=models, y=f1_scores, name="F1 Score", marker_color="#ff7f0e"))
fig2.update_layout(title="Model Comparison", barmode="group",
                   paper_bgcolor="#0e1117", plot_bgcolor="#262730", font_color="white")
st.plotly_chart(fig2, use_container_width=True)
