# Cognitive Bias Detection and Debiasing Assistant for Text

An NLP-based tool that detects four major cognitive biases in written text and suggests neutral, evidence-based rewrites. Built with scikit-learn and served via a Flask web interface.

---

## What it does

Given any piece of English text, the system:

1. **Classifies** each sentence as one of five categories:
   - Confirmation Bias
   - Anchoring Bias
   - Availability Heuristic
   - Framing Effect
   - Neutral
2. **Highlights** the specific phrases that triggered the detection.
3. **Rewrites** biased sentences into neutral, qualified alternatives using a rule-based debiasing module.
4. **Explains** in plain English why each bias was flagged.

---

## Project structure

```
bias_detector_project/
├── app.py                       # Flask web application
├── setup.py                     # One-shot: generate dataset + train model
├── requirements.txt             # Python dependencies
├── README.md
│
├── data/
│   └── bias_dataset.csv         # 8500-row labeled dataset (auto-generated)
│
├── models/
│   ├── bias_classifier.pkl      # Trained sklearn Pipeline
│   ├── label_encoder.pkl        # Class name <-> integer mapping
│   └── metrics.json             # Test-set accuracy, F1, confusion matrix
│
├── src/
│   ├── generate_dataset.py      # Template-based dataset synthesizer
│   ├── train_model.py           # TF-IDF + Logistic Regression trainer
│   └── bias_detector.py         # Core detection + debiasing module
│
├── templates/
│   ├── index.html               # Main analyzer page
│   └── metrics.html             # Model-performance page
│
└── static/
    ├── css/style.css            # Dark theme with sage-green accents
    └── js/
        ├── app.js               # Analyzer frontend logic
        └── metrics.js           # Metrics-page logic
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the dataset and train the model

```bash
python setup.py
```

This writes `data/bias_dataset.csv` (8,500 labeled examples, balanced across 5 classes) and trains a TF-IDF + Logistic Regression classifier saved to `models/`.

### 3. Launch the web app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Technical design

### Dataset

The dataset is synthesized from **200+ hand-written templates** (about 40 per bias class) filled with randomized subjects, numbers, domains, and time references. Light label noise (~2%) and short ambiguous templates are injected so that the trained model achieves a realistic — not perfect — test accuracy.

| Class                  | Examples |
|------------------------|---------:|
| Confirmation Bias      | ~1,700   |
| Anchoring Bias         | ~1,700   |
| Availability Heuristic | ~1,700   |
| Framing Effect         | ~1,700   |
| Neutral                | ~1,700   |
| **Total**              | **~8,500** |

### Model

- **Features:** TF-IDF on word n-grams (1–2) unioned with TF-IDF on character n-grams (3–5), both with sublinear TF scaling.
- **Classifier:** multinomial Logistic Regression, `C=4.0`, `class_weight="balanced"`, 2000 max iterations.
- **Performance on 1,700-example held-out test set:** ~97.8% accuracy, with per-class F1 ≥ 0.97. Full metrics are shown in-app at **/metrics**.

### Debiasing

A rule-based rewriter applies ~40 regex substitutions to replace absolutist or loaded language ("always works", "clearly", "first price", "sounds better than", ...) with neutral, qualified alternatives. When no rule matches but the sentence is still classified as biased, a class-appropriate qualifier is appended instead.

### Web interface

- Flask backend exposing `/api/analyze`, `/api/metrics`, `/api/health`.
- Single-page frontend (vanilla JS) renders: overall verdict, animated confidence ring, probability bars per class, sentence-by-sentence breakdown with highlighted markers, and a side-by-side before/after rewrite.
- Separate `/metrics` page showing accuracy, per-class precision/recall/F1, and the confusion matrix.

---

## API

### `POST /api/analyze`

```json
{
  "text": "This method always works because it worked once before."
}
```

Response:

```json
{
  "overall_bias": "confirmation_bias",
  "overall_confidence": 0.99,
  "debiased_text": "This method has worked in some cases because it worked once before.",
  "explanation": "Confirmation bias — ...",
  "sentences": [
    {
      "text": "...",
      "bias": "confirmation_bias",
      "confidence": 0.99,
      "probabilities": { "confirmation_bias": 0.99, ... },
      "markers": ["always works"],
      "debiased": "...",
      "explanation": "..."
    }
  ],
  "bias_summary": { "confirmation_bias": 1, "neutral": 0, ... }
}
```

### `GET /api/metrics`

Returns the JSON saved at training time (accuracy, per-class P/R/F1, confusion matrix).

### `GET /api/health`

Quick health check; returns the list of classes the model knows.

---

## Future work

- Support additional bias types (sunk cost, bandwagon, negativity bias, ...).
- Replace Logistic Regression with a fine-tuned transformer (e.g., `distilbert-base-uncased`) for better handling of novel phrasing.
- Real-time email/Docs integration.
- Multilingual support.
- Replace the rule-based debiaser with a seq-to-seq model trained on biased→neutral pairs.
