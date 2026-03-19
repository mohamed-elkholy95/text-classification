<div align="center">

# 🏷️ Text Classification System

**Multi-model text classification** with TF-IDF, embeddings, transformer, and ensemble methods

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-64%20passed-success?style=flat-square)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square)](https://streamlit.io)

</div>

## Overview

A **comprehensive text classification pipeline** supporting multiple feature extraction methods (TF-IDF, word embeddings), several classifiers (SVM, Logistic Regression, Naive Bayes), a transformer-based model, and model ensembling — all with systematic evaluation and hyperparameter tuning.

## Features

- 📊 **TF-IDF Features** — Configurable n-gram range, min/max document frequency
- 🧮 **Word Embeddings** — SVD-based dense representations from corpus
- 🏆 **5 Classifiers** — SVM, Logistic Regression, Naive Bayes, Random Forest, Transformer
- 🤝 **Model Ensemble** — Weighted voting and stacking ensemble strategies
- 📈 **Data Augmentation** — Synonym replacement, back-translation simulation, and random augmentation
- 🔧 **Hyperparameter Tuning** — Grid search with cross-validation
- ✅ **64 Tests** — Full coverage of all pipeline components

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/text-classification.git
cd text-classification
pip install -r requirements.txt
python -m pytest tests/ -v
streamlit run streamlit_app/app.py
```

## Project Structure

```
├── src/
│   ├── data/          # Dataset loader, preprocessor, augmentor
│   ├── features/      # TF-IDF, embeddings, feature combiner
│   ├── models/        # Baseline models, transformer, ensemble
│   ├── evaluation.py  # Metrics and reporting
│   └── api/main.py    # FastAPI endpoints
├── streamlit_app/pages/  # 4 dashboard pages
├── tests/                 # 64 tests
└── requirements.txt
```

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
