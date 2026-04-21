"""
Script 2: Baseline ML Pipeline — TF-IDF + Logistic Regression
==============================================================
Trains a strong, interpretable baseline classifier on the cleaned dataset.
Performs stratified train/test split, reports full metrics, and saves the
serialised pipeline to models/baseline_pipeline.pkl

Usage:
    python scripts/2_train_baseline.py --data data/cleaned_data.csv
"""

import argparse
import os
import json
import pickle
import re
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources once
for resource in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


# ─── Text Preprocessing ───────────────────────────────────────────────────────
_stop = set(stopwords.words("english"))
_stemmer = PorterStemmer()

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, remove URLs/mentions, expand hashtags, strip punctuation, stem."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)          # keep hashtag words
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [_stemmer.stem(t) for t in tokens if t not in _stop and len(t) > 1]
    return " ".join(tokens)


# ─── Build Pipeline ───────────────────────────────────────────────────────────
def build_logreg_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=100_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )),
    ])


def build_svm_pipeline() -> Pipeline:
    """Alternative: LinearSVC with Platt calibration for probability output."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=100_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced", max_iter=2000, random_state=42),
            cv=3,
        )),
    ])


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(pipe: Pipeline, X_test: list, y_test: np.ndarray, label: str, out_dir: str):
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {label} — Test Set Metrics")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['non-harmful','harmful'])}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["non-harmful", "harmful"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {label}")
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"cm_{label.replace(' ', '_').lower()}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {plot_path}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ─── Cross-Validation ─────────────────────────────────────────────────────────
def cross_validate(pipe: Pipeline, X: list, y: np.ndarray, label: str):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1", n_jobs=-1)
    print(f"\n  5-Fold CV F1 ({label}): {scores.mean():.4f} ± {scores.std():.4f}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train baseline TF-IDF classifier")
    parser.add_argument("--data",    default="data/cleaned_data.csv")
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading data from {args.data} …")
    df = pd.read_csv(args.data)
    print(f"  Total samples : {len(df):,}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    print("[2/5] Cleaning text …")
    df["clean_text"] = df["tweet"].apply(clean_text)

    X = df["clean_text"].tolist()
    y = df["binary_label"].values

    # ── Split ─────────────────────────────────────────────────────────────────
    print("[3/5] Stratified train/test split …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Train Logistic Regression ─────────────────────────────────────────────
    print("[4/5] Training Logistic Regression …")
    lr_pipe = build_logreg_pipeline()
    lr_pipe.fit(X_train, y_train)
    cross_validate(lr_pipe, X_train, y_train, "Logistic Regression (train)")
    lr_metrics = evaluate(lr_pipe, X_test, y_test, "Logistic Regression", args.out_dir)

    # ── Train LinearSVC ───────────────────────────────────────────────────────
    print("[4b/5] Training LinearSVC …")
    svm_pipe = build_svm_pipeline()
    svm_pipe.fit(X_train, y_train)
    svm_metrics = evaluate(svm_pipe, X_test, y_test, "LinearSVC", args.out_dir)

    # ── Choose best model ─────────────────────────────────────────────────────
    best_pipe, best_name, best_metrics = (
        (lr_pipe,  "Logistic Regression", lr_metrics)
        if lr_metrics["f1"] >= svm_metrics["f1"]
        else (svm_pipe, "LinearSVC", svm_metrics)
    )
    print(f"\n  Best model: {best_name} (F1={best_metrics['f1']:.4f})")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("[5/5] Saving model …")
    model_path = os.path.join(args.out_dir, "baseline_pipeline.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_pipe, f)
    print(f"  Saved → {model_path}")

    metrics_path = os.path.join(args.out_dir, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"model": best_name, **best_metrics}, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    print("\n✅ Baseline training complete.")


if __name__ == "__main__":
    main()
