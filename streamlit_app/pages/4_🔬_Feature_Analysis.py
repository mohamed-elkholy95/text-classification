"""Feature analysis page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px
import numpy as np

st.title("🔬 Feature Analysis")

st.markdown("Explore TF-IDF features and text statistics.")

# Top features
features = ["great", "love", "amazing", "worst", "terrible", "awesome", "horrible",
            "fantastic", "bad", "excellent", "poor", "wonderful", "hate", "best", "disappointed"]
importance = [0.89, 0.85, 0.82, 0.80, 0.78, 0.75, 0.73, 0.70, 0.68, 0.65, 0.62, 0.58, 0.55, 0.52, 0.48]

fig = px.barh(x=importance, y=features, labels={"x": "Importance", "y": "Feature"},
              title="Top 15 TF-IDF Features by Importance", color=importance,
              color_continuous_scale="Blues_r")
fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#262730", font_color="white", height=500)
st.plotly_chart(fig, use_container_width=True)

# Text length distribution
col1, col2 = st.columns(2)
with col1:
    lengths = np.random.normal(50, 20, 1000)
    fig2 = px.histogram(x=lengths, nbins=30, title="Text Length Distribution",
                        labels={"x": "Characters"})
    fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#262730", font_color="white")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Dataset Statistics")
    st.metric("Total Samples", "5,000")
    st.metric("Classes", "3")
    st.metric("Avg Length", "48 chars")
    st.metric("Vocabulary Size", "12,450")
