<div align="center">

# 🏷️ Text Classification System

**Multi-model text classification** with TF-IDF, word embeddings, transformers, and ensemble methods — built as an educational portfolio project.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?style=flat-square)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-passing-success?style=flat-square)](#-testing)

</div>

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      TEXT CLASSIFICATION PIPELINE                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │ Raw Data │───▶│  Preprocess  │───▶│  Feature Extract   │    │
│  │ (CSV/    │    │  - Clean     │    │  - TF-IDF          │    │
│  │  Synthetic)│  │  - Tokenize  │    │  - Embeddings      │    │
│  └──────────┘    │  - Encode    │    │  - Combined        │    │
│                  └──────────────┘    └────────┬───────────┘    │
│                                               │                 │
│                  ┌──────────────┐    ┌────────▼───────────┐    │
│                  │  Evaluation  │◀───│  Classifiers       │    │
│                  │  - Metrics   │    │  - Naive Bayes     │    │
│                  │  - Curves    │    │  - Logistic Reg    │    │
│                  │  - Calibratn │    │  - SVM             │    │
│                  │  - Report    │    │  - Random Forest   │    │
│                  └──────┬───────┘    │  - Transformer     │    │
│                         │            │  - Ensemble        │    │
│                  ┌──────▼───────┐    └────────────────────┘    │
│                  │   Serving   │                               │
│                  │  - FastAPI  │    ┌────────────────────┐    │
│                  │  - Streamlit│◀───│  Hyperparameter    │    │
│                  └─────────────┘    │  Tuning (Grid)     │    │
│                                     └────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Cross-Validation │ Data Augmentation │ Model Ensembling│  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🧠 Key Concepts

| Concept | Description |
|---------|-------------|
| **TF-IDF** | Term Frequency–Inverse Document Frequency converts text into numerical vectors. Words that appear often in one document but rarely across the corpus get high scores, making them discriminative features. |
| **Naive Bayes** | Applies Bayes' theorem with a "naive" independence assumption between features. Despite the assumption rarely holding, it works remarkably well for text classification because word co-occurrence patterns still encode class information. |
| **SVM (Support Vector Machine)** | Finds the optimal hyperplane that maximizes the margin between classes. Linear SVMs are especially effective for high-dimensional sparse data like TF-IDF vectors. |
| **Logistic Regression** | Models the probability of class membership using a logistic function. Coefficients are directly interpretable as feature importance — perfect for understanding *why* the model makes each prediction. |
| **Ensemble Methods** | Combine multiple weak learners into a stronger predictor. Voting averages predictions; stacking trains a meta-learner on base model outputs to capture complementary strengths. |
| **Confidence Calibration** | Raw model scores aren't always true probabilities. Platt scaling (sigmoid) and isotonic regression transform scores so that "70% confident" actually means ~70% accuracy. |
| **Cross-Validation** | Stratified k-fold CV splits data into k folds while preserving class ratios, giving a lower-variance estimate of generalization performance than a single train/test split. |

> 📖 See [`docs/CONCEPTS.md`](docs/CONCEPTS.md) for deeper explanations of each concept.

## 📁 Project Structure

```
text-classification/
├── src/
│   ├── __init__.py
│   ├── config.py                    # All configuration constants
│   ├── evaluation.py                # Metrics, confusion matrix, reporting
│   ├── calibration.py               # Platt scaling & isotonic regression
│   ├── cross_validation.py          # Stratified k-fold CV
│   ├── learning_curves.py           # Bias-variance diagnostics
│   ├── model_comparison.py          # Side-by-side model benchmarking
│   ├── text_analyzer.py             # Corpus-level text statistics
│   ├── tuning.py                    # Grid search hyperparameter tuning
│   ├── data/
│   │   ├── dataset_loader.py        # Synthetic data & CSV loaders
│   │   ├── preprocessor.py          # Text cleaning, tokenization, label encoding
│   │   └── augmentor.py             # Synonym replacement, random augmentation
│   ├── features/
│   │   ├── tfidf_features.py        # TF-IDF vectorizer with configurable n-grams
│   │   ├── embedding_features.py    # SVD-based dense word embeddings
│   │   └── feature_combiner.py      # Merge TF-IDF + embedding features
│   ├── models/
│   │   ├── baseline_models.py       # NB, LR, SVM, Random Forest
│   │   ├── model_ensemble.py        # Voting & stacking ensembles
│   │   └── transformer_classifier.py # DistilBERT fine-tuning
│   └── api/
│       ├── main.py                  # FastAPI application
│       └── models.py                # Pydantic request/response schemas
├── streamlit_app/
│   ├── app.py                       # Main dashboard (multipage)
│   └── pages/
│       ├── 1_📊_Overview.py          # Dataset stats & class distribution
│       ├── 2_💬_Classify.py           # Live text classification demo
│       ├── 3_📈_Training_Metrics.py   # Model performance charts
│       └── 4_🔬_Feature_Analysis.py  # TF-IDF importance & vocabulary stats
├── tests/                            # 70+ tests with pytest
├── docs/
│   ├── CONCEPTS.md                  # Educational deep-dives
│   └── PROJECT_PLAN.md              # Development roadmap
├── train_distilbert_agnews.py        # Fine-tune DistilBERT on AG News
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Clone and navigate
git clone https://github.com/mohamed-elkholy95/text-classification.git
cd text-classification

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests
python -m pytest tests/ -v

# 5. Launch the Streamlit dashboard
streamlit run streamlit_app/app.py

# 6. (Optional) Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8002
```

## 🔌 API Endpoints

The FastAPI server exposes a REST API for programmatic text classification.

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|-------------|----------|
| `POST` | `/predict` | Classify a single text | `{"text": "..."}` | `{"label": 1, "confidence": 0.92, ...}` |
| `POST` | `/predict_batch` | Classify multiple texts | `{"texts": [...]}` | `{"predictions": [...]}` |
| `GET` | `/health` | Health check | — | `{"status": "ok"}` |
| `GET` | `/models` | List available models | — | `{"models": ["nb", "lr", "svm", ...]}` |
| `GET` | `/metrics` | Latest evaluation metrics | — | Metrics dict |

```bash
# Example: classify a review
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'
```

## 📊 Streamlit Dashboard

The interactive dashboard provides four pages:

| Page | Description |
|------|-------------|
| **📊 Overview** | Dataset statistics, class distribution, sample texts |
| **💬 Classify** | Enter any text and see predictions from all models side-by-side |
| **📈 Training Metrics** | Accuracy, F1, precision/recall curves, ROC curves |
| **🔬 Feature Analysis** | Real TF-IDF importance from Logistic Regression coefficients, word frequencies, vocabulary statistics |

```bash
streamlit run streamlit_app/app.py
```

The dashboard uses a dark theme and `@st.cache_resource` for efficient data loading and model caching.

## ✅ Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_evaluation.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

The test suite covers:
- **Data pipeline**: dataset loading, preprocessing, augmentation
- **Feature extraction**: TF-IDF, embeddings, feature combining
- **Models**: baseline classifiers, ensemble, transformer
- **Evaluation**: metrics (binary + multiclass), confusion matrix, per-class metrics, edge cases
- **Advanced**: calibration, cross-validation, learning curves
- **API**: endpoint validation with TestClient

## 📚 What You'll Learn

Building this project teaches core ML concepts through hands-on implementation:

1. **Feature Engineering for Text** — Why TF-IDF outperforms raw bag-of-words, how n-grams capture context, and when dense embeddings are worth the cost.

2. **Model Selection & Comparison** — When to use Naive Bayes (fast, good baselines) vs. SVM (strong margins) vs. Logistic Regression (interpretable coefficients) vs. ensemble methods (best accuracy).

3. **Evaluation Beyond Accuracy** — Why F1 matters for imbalanced classes, how to read precision-recall curves, and why a confusion matrix reveals failure modes that aggregate metrics hide.

4. **Confidence Calibration** — Why raw model scores are not true probabilities, how Platt scaling and isotonic regression fix this, and when calibration matters (thresholding, risk-sensitive decisions).

5. **Bias-Variance Diagnosis** — How learning curves reveal underfitting (high bias) vs. overfitting (high variance), and how to fix each (more features vs. more data/regularization).

6. **Cross-Validation** — Why stratified k-fold gives more reliable estimates than a single split, and how to use it for model selection without leaking test data.

7. **Ensemble Methods** — Why combining diverse models reduces variance, how voting vs. stacking differ, and when ensembles are worth the complexity.

8. **Production Considerations** — API design with FastAPI, Streamlit dashboards for non-technical stakeholders, and how to cache expensive computations.

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
