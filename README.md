# Cognitive Bias Detection and Debiasing Assistant for Text

An NLP-based tool that detects four major cognitive biases in written text and suggests neutral, evidence-based rewrites. Built with scikit-learn and served via a Flask web interface.

## What it does

Given any piece of English text, the system classifies each sentence as one of five categories (Confirmation Bias, Anchoring Bias, Availability Heuristic, Framing Effect, or Neutral), highlights the specific phrases that triggered the detection, rewrites biased sentences into neutral alternatives, and explains why each bias was flagged.

## Quick start

    git clone https://github.com/Bhavya0412/cognitive-bias-detector.git
    cd cognitive-bias-detector
    pip install -r requirements.txt
    python app.py

Then open http://localhost:5000 in your browser.

To regenerate the dataset and retrain the model from scratch:

    python setup.py
    python app.py

## Project structure

    cognitive-bias-detector/
    ├── app.py                   # Flask web application
    ├── setup.py                 # One-shot: generate dataset + train model
    ├── requirements.txt
    ├── README.md
    ├── data/
    │   └── bias_dataset.csv     # 8,500-row labeled dataset
    ├── models/
    │   ├── bias_classifier.pkl  # Trained sklearn Pipeline
    │   ├── label_encoder.pkl
    │   └── metrics.json
    ├── src/
    │   ├── generate_dataset.py
    │   ├── train_model.py
    │   └── bias_detector.py
    ├── templates/
    │   ├── index.html
    │   └── metrics.html
    └── static/
        ├── css/style.css
        └── js/
            ├── app.js
            └── metrics.js

## Technical design

- Dataset: 8,500 labeled examples synthesized from 200+ hand-written templates, balanced across 5 classes, with ~2% label noise.
- Features: TF-IDF on word n-grams (1-2) unioned with TF-IDF on character n-grams (3-5).
- Classifier: Multinomial Logistic Regression, class_weight=balanced.
- Test accuracy: ~97.8% on a 1,700-example held-out test set.
- Debiasing: ~40 regex substitution rules that replace absolutist or loaded phrasing with neutral alternatives.

## API

- GET  /              — Main analyzer page
- GET  /metrics       — Model-performance page
- POST /api/analyze   — JSON in/out: classifies and rewrites input text
- GET  /api/metrics   — Returns training metrics JSON
- GET  /api/health    — Health check