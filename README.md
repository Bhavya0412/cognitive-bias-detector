# HarmGuard — NLP Detection Suite

A comprehensive NLP toolkit featuring data-driven hate speech detection and cognitive bias analysis. This repository combines a robust ML pipeline with real-time web interfaces and browser extensions.

---

## 1. HarmGuard: Hate Speech Detection
A data-driven NLP pipeline that detects hate speech and offensive language using machine learning, served through a real-time web interface and Chrome extension.

### Architecture
```
NLP/
├── scripts/
│   ├── 1_validate_data.py  ← EDA, null/duplicate/balance checks
│   ├── 2_train_baseline.py ← TF-IDF + Logistic Regression pipeline
│   └── 3_train_bert.py     ← DistilBERT fine-tuning (GPU-ready)
├── api/
│   ├── app.py              ← FastAPI backend + serves web UI
│   └── ui.html             ← Standalone web interface
├── extension/
│   ├── manifest.json       ← Chrome Extension (Manifest V3)
│   ├── background.js
│   ├── content.js
│   ├── popup.html
│   └── popup.js
└── data/                   ← Place labeled_data.csv here (gitignored)
```

### Quickstart (HarmGuard)
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run the pipeline**: 
   - `python scripts/1_validate_data.py`
   - `python scripts/2_train_baseline.py`
   - `python scripts/3_train_bert.py`
3. **Start the API**: `uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload`
4. Open **http://localhost:8000**

---

## 2. Cognitive Bias Detection
An NLP-based tool that detects four major cognitive biases (Confirmation, Anchoring, Availability Heuristic, Framing Effect) and suggests neutral rewrites.

### Quickstart (Bias Detector)
1. **Build dataset and train**: `python setup.py`
2. **Launch Flask app**: `python app.py`
3. Open **http://localhost:5000**

---

## Technical Design & Performance

### HarmGuard (Hate Speech)
- **Dataset**: Davidson et al. (2017) corpus (~25k tweets).
- **Models**: TF-IDF + Logistic Regression (Baseline) and DistilBERT (Fine-tuned).
- **Performance**: ~92% Accuracy with BERT.

### Bias Detector
- **Dataset**: Synthesized from 200+ hand-written templates (~8,500 examples).
- **Model**: TF-IDF (word/char n-grams) + Logistic Regression.
- **Performance**: ~97.8% Accuracy on balanced test set.

---

## Chrome Extension
1. Open `chrome://extensions/`
2. Enable **Developer Mode**
3. Click **Load unpacked** → select the `extension/` folder
4. Ensure the HarmGuard API is running at `localhost:8000`
