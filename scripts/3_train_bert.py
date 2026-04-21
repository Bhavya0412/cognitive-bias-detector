"""
Script 3: Fine-tune DistilBERT for Hate/Offensive Speech Detection
==================================================================
Usage:
    python scripts/3_train_bert.py --data data/cleaned_data.csv

Outputs:
    models/bert/   — HuggingFace model weights
    models/bert_metrics.json
"""

import argparse, os, json, random, time
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 4
LR         = 2e-5

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts; self.labels = labels; self.tokenizer = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=MAX_LEN, padding="max_length",
                              truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)}


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        optimizer.zero_grad()
        out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total_loss += out.loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        preds = model(input_ids=ids, attention_mask=mask).logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds); all_labels.extend(batch["labels"].numpy())
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    return acc, prec, rec, f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/cleaned_data.csv")
    parser.add_argument("--out_dir",    default="models/bert")
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n  Device: {DEVICE.upper()}")

    df     = pd.read_csv(args.data)
    texts  = df["tweet"].astype(str).tolist()
    labels = df["binary_label"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=SEED)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

    train_loader = DataLoader(TweetDataset(X_train, y_train, tokenizer),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(TweetDataset(X_test,  y_test,  tokenizer),
                              batch_size=args.batch_size, shuffle=False, num_workers=0)

    total_steps  = len(train_loader) * args.epochs
    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)

    best_f1 = 0.0; history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        acc, prec, rec, f1, _, _ = eval_epoch(model, test_loader, DEVICE)
        history.append({"epoch": epoch, "loss": loss, "f1": f1})
        print(f"  Ep {epoch}/{args.epochs}  loss={loss:.4f}  acc={acc:.4f}  F1={f1:.4f}  ({time.time()-t0:.0f}s)")
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
            print(f"    ✔ Best saved (F1={best_f1:.4f})")

    best_model = DistilBertForSequenceClassification.from_pretrained(args.out_dir).to(DEVICE)
    acc, prec, rec, f1, preds, labels_true = eval_epoch(best_model, test_loader, DEVICE)
    print(classification_report(labels_true, preds, target_names=["non-harmful","harmful"]))

    metrics_path = os.path.join(os.path.dirname(args.out_dir), "bert_metrics.json")
    with open(metrics_path, "w") as mf:
        json.dump({"model": MODEL_NAME, "accuracy": acc, "precision": prec,
                   "recall": rec, "f1": f1, "history": history}, mf, indent=2)
    print(f"\n✅ Done. Metrics → {metrics_path}")


if __name__ == "__main__":
    main()
