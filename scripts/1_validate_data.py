"""
Script 1: Data Validation & EDA
================================
Validates and cleans the labeled_data.csv (Davidson et al. hate speech dataset).
Checks for nulls, duplicates, class imbalance, and text length distribution.
Outputs a cleaned file to data/cleaned_data.csv.

Usage:
    python scripts/1_validate_data.py --input data/labeled_data.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "hate_speech", 1: "offensive_language", 2: "neither"}
BINARY_MAP = {0: 1, 1: 1, 2: 0}  # hate/offensive → 1,  neutral → 0
TEXT_COL = "tweet"
LABEL_COL = "class"
MIN_TEXT_LEN = 5  # characters


def load_data(path: str) -> pd.DataFrame:
    print(f"\n[1/5] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    return df


def check_nulls(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/5] Checking for null values …")
    null_counts = df.isnull().sum()
    print(null_counts.to_string())
    before = len(df)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    after = len(df)
    print(f"  Dropped {before - after} rows with nulls in key columns.")
    return df


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/5] Checking for duplicates …")
    n_dups = df.duplicated(subset=[TEXT_COL]).sum()
    print(f"  Found {n_dups} duplicate tweets.")
    df = df.drop_duplicates(subset=[TEXT_COL])
    print(f"  Shape after dedup: {df.shape}")
    return df


def check_text_quality(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/5] Checking text quality …")
    df["text_len"] = df[TEXT_COL].astype(str).str.len()
    short_mask = df["text_len"] < MIN_TEXT_LEN
    print(f"  Rows with fewer than {MIN_TEXT_LEN} chars: {short_mask.sum()}")
    df = df[~short_mask].copy()
    print(f"  Mean text length: {df['text_len'].mean():.1f} chars")
    print(f"  Max text length : {df['text_len'].max()} chars")
    return df


def analyse_class_balance(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/5] Analysing class balance …")

    # Original 3-class
    counts = df[LABEL_COL].value_counts().sort_index()
    for cls, cnt in counts.items():
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {cnt:,} samples ({cnt/len(df)*100:.1f}%)")

    # Binary mapping
    df["binary_label"] = df[LABEL_COL].map(BINARY_MAP)
    bin_counts = df["binary_label"].value_counts().sort_index()
    print("\n  Binary label distribution:")
    labels = {0: "non-harmful", 1: "harmful"}
    for cls, cnt in bin_counts.items():
        print(f"  {cls} ({labels[cls]}): {cnt:,} samples ({cnt/len(df)*100:.1f}%)")

    imbalance_ratio = bin_counts[1] / bin_counts[0] if bin_counts[0] else float("inf")
    print(f"\n  Imbalance ratio (harmful:non-harmful): {imbalance_ratio:.2f}")
    if imbalance_ratio > 3 or imbalance_ratio < 0.33:
        print("  ⚠  Significant class imbalance detected — consider SMOTE or class_weight='balanced'")
    return df


def plot_eda(df: pd.DataFrame, out_dir: str = "data"):
    os.makedirs(out_dir, exist_ok=True)
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Dataset EDA", fontsize=16, fontweight="bold")

    # Class distribution
    class_labels = [CLASS_NAMES[c] for c in sorted(df[LABEL_COL].unique())]
    counts = [df[df[LABEL_COL] == c].shape[0] for c in sorted(df[LABEL_COL].unique())]
    axes[0].bar(class_labels, counts, color=["#e74c3c", "#f39c12", "#2ecc71"])
    axes[0].set_title("Original 3-Class Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

    # Binary label distribution
    bin_counts = df["binary_label"].value_counts().sort_index()
    axes[1].pie(bin_counts, labels=["Non-harmful", "Harmful"],
                autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"], startangle=90)
    axes[1].set_title("Binary Label Distribution")

    # Text length distribution
    axes[2].hist(df["text_len"], bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    axes[2].set_title("Tweet Length Distribution (chars)")
    axes[2].set_xlabel("Length")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "eda_plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  EDA plot saved → {out_path}")
    plt.close()


def save_cleaned(df: pd.DataFrame, out_dir: str = "data"):
    # Keep only necessary columns
    keep_cols = [TEXT_COL, LABEL_COL, "binary_label"]
    df = df[keep_cols].copy()
    out_path = os.path.join(out_dir, "cleaned_data.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Cleaned dataset saved → {out_path}  ({len(df):,} rows)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Validate and clean the hate-speech dataset")
    parser.add_argument("--input", default="data/labeled_data.csv",
                        help="Path to raw labeled_data.csv")
    parser.add_argument("--out_dir", default="data",
                        help="Directory to save cleaned data and plots")
    args = parser.parse_args()

    df = load_data(args.input)
    df = check_nulls(df)
    df = check_duplicates(df)
    df = check_text_quality(df)
    df = analyse_class_balance(df)
    plot_eda(df, out_dir=args.out_dir)
    save_cleaned(df, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
