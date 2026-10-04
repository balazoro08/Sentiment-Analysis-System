# Sentiment Analysis System

An end-to-end NLP and Machine Learning application that detects **Positive**, **Negative**, and **Neutral** sentiment across product reviews, social media comments, app feedback, and customer support tickets. Built with a Python FastAPI backend and a modern glassmorphism web UI.

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-emerald.svg)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

---

## ✨ Features

- **Real-Time Live Analyzer**: Analyze individual text reviews with instant sentiment diagnosis, probability breakdown progress bars, polarity index (-1.0 to +1.0), and subjectivity scores.
- **Word-Level Sentiment Driver Inspector**: Visual token highlighter showing which exact words driven positive (green) or negative (red) classification.
- **Negation-Aware NLP Preprocessing**: Intelligently handles complex phrasing (e.g., *"not bad"*, *"never fails"* vs *"not good"*) to prevent false negative triggers.
- **Batch CSV / JSON Processor**: Upload custom CSV datasets or load sample benchmarks for batch sentiment scoring, complete with summary statistics, distribution pie charts, confidence histograms, and searchable data tables.
- **Model Analytics & Confusion Matrix**: Visual 3x3 confusion matrix grid, F1/precision/recall metrics, top positive/negative feature weights, and one-click model retraining.
- **Interactive API Playground**: Built-in REST API runner with instant code snippet generators for Python `requests`, JavaScript `fetch`, and cURL commands.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.13, FastAPI, Uvicorn, Pydantic
- **Machine Learning & NLP**: Scikit-Learn (TF-IDF Vectorizer + Logistic Regression Classifier), NLTK, Pandas, NumPy
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism & dark theme), Vanilla JavaScript (ES6+), Chart.js
- **Testing**: Python `unittest`

---

---

## 📂 Project Structure

```
Sentiment-Analysis-System/
├── app.py                  # FastAPI REST Server & Static Asset Router
├── sentiment_engine.py     # NLP Preprocessor, TF-IDF Classifier & Token Highlighter
├── dataset_generator.py    # Multi-domain Sentiment Datasets & Sample CSV Generator
├── test_system.py          # Automated Unit Test Suite
├── requirements.txt        # Python Project Dependencies
├── README.md               # Documentation
└── static/
    ├── index.html          # Web UI Dashboard HTML
    ├── css/styles.css      # Custom Dark Glassmorphism CSS Design System
    └── js/app.js           # Frontend Interactive Controller & Chart.js Integration
```

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more details.
