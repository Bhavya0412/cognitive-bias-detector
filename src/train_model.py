"""
Train a cognitive-bias classifier on the synthetic dataset.

Pipeline:
    TF-IDF (word n-grams + char n-grams) -> Logistic Regression
Saves:
    models/bias_classifier.pkl  - full sklearn Pipeline
    models/label_encoder.pkl    - LabelEncoder (for class name lookup)
    models/metrics.json         - accuracy, per-class F1, confusion matrix
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "bias_dataset.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def build_pipeline():
    """Word + char n-gram TF-IDF fused into a single feature space."""
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    features = FeatureUnion([("word", word_vec), ("char", char_vec)])
    clf = LogisticRegression(
        max_iter=2000,
        C=4.0,
        class_weight="balanced",
        n_jobs=-1,
    )
    return Pipeline([("features", features), ("clf", clf)])


def main():
    print(f"Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows, {df['label'].nunique()} classes")
    print(df["label"].value_counts().to_string())

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X = df["text"].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

    pipe = build_pipeline()
    print("\nFitting pipeline (TF-IDF word+char -> Logistic Regression)...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")

    report_str = classification_report(
        y_test, y_pred, target_names=le.classes_, digits=4
    )
    print("\nClassification report:")
    print(report_str)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows = true, cols = predicted):")
    print("Classes:", list(le.classes_))
    print(cm)

    # Save artifacts
    with open(os.path.join(MODEL_DIR, "bias_classifier.pkl"), "wb") as f:
        pickle.dump(pipe, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    report_dict = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )
    metrics = {
        "accuracy": float(acc),
        "classes": list(le.classes_),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            cls: {
                "precision": float(report_dict[cls]["precision"]),
                "recall": float(report_dict[cls]["recall"]),
                "f1": float(report_dict[cls]["f1-score"]),
                "support": int(report_dict[cls]["support"]),
            }
            for cls in le.classes_
        },
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {MODEL_DIR}/bias_classifier.pkl")
    print(f"Saved label encoder to {MODEL_DIR}/label_encoder.pkl")
    print(f"Saved metrics to {MODEL_DIR}/metrics.json")


if __name__ == "__main__":
    main()
