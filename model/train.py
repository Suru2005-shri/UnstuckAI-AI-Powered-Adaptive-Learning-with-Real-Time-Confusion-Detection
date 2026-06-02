"""
train.py
========
Training script for UnstuckAI confusion detection models.

Usage:
    # Stage 1 — fewer than 500 confusion events
    python model/train.py --model rf --data data/windows.csv

    # Stage 2 — 500+ confusion events (primary model)
    python model/train.py --model lstm --data data/windows.csv \
        --pretrained model/checkpoints/pretrained_encoder.pt

    # Stage 3 — 2000+ confusion events
    python model/train.py --model transformer --data data/windows.csv

Outputs:
    model/checkpoints/best.pt           best checkpoint (LSTM/Transformer)
    model/checkpoints/rf_model.pkl      trained RF model
    model/checkpoints/train_history.csv epoch-by-epoch metrics
"""

import argparse
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.model   import build_model, ConfusionRF
from model.dataset import make_loaders, WindowSequenceDataset
from model.features import FEATURE_COLS, load_norm_stats, apply_norm


# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────

HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 64
LR           = 1e-3
POS_WEIGHT   = 3.0      # upweight confused class (rare events)
PATIENCE     = 8        # early stopping


# ─────────────────────────────────────────────
# Epoch runner
# ─────────────────────────────────────────────

def run_epoch(model, loader, optimizer, loss_fn, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = model(X_batch)
            loss = loss_fn(logits, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            total_loss += loss.item() * len(y_batch)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_batch.long().cpu().numpy())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    f1       = f1_score(all_labels, all_preds, zero_division=0)
    auc      = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    fpr      = _false_positive_rate(all_labels, all_preds)
    return avg_loss, f1, auc, fpr


def _false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / max(fp + tn, 1)


# ─────────────────────────────────────────────
# LSTM / Transformer training
# ─────────────────────────────────────────────

def train_deep(args, device):
    norm_path = os.path.join(os.path.dirname(args.data), "norm_stats.json")
    loaders, _ = make_loaders(args.data, norm_path, BATCH_SIZE)

    model = build_model(
        args.model,
        input_size  = len(FEATURE_COLS),
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
    ).to(device)

    # Load EEG pre-trained encoder if available
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"[pretrain] Loading encoder weights from {args.pretrained}")
        model.load_pretrained_encoder(args.pretrained, device=str(device))
    elif args.pretrained:
        print(f"[pretrain] Checkpoint not found at {args.pretrained} — training from scratch")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.model.upper()} — {total_params:,} trainable parameters")

    pos_weight = torch.tensor([POS_WEIGHT]).to(device)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss  = float("inf")
    patience_count = 0
    history        = []
    ckpt_path      = os.path.join(args.checkpoint_dir, "best.pt")

    print("\n" + "─" * 78)
    print(f"{'Ep':>4}  {'TrLoss':>7}  {'TrF1':>6}  {'VaLoss':>7}  "
          f"{'VaF1':>6}  {'VaAUC':>6}  {'FPR':>5}  {'LR':>8}")
    print("─" * 78)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, tr_auc, _      = run_epoch(model, loaders["train"], optimizer, loss_fn, device, True)
        val_loss, val_f1, val_auc, fpr = run_epoch(model, loaders["val"],   optimizer, loss_fn, device, False)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        flag = " *" if val_loss < best_val_loss else ""
        print(f"{epoch:>4}  {tr_loss:>7.4f}  {tr_f1:>6.4f}  {val_loss:>7.4f}  "
              f"{val_f1:>6.4f}  {val_auc:>6.4f}  {fpr:>5.3f}  {lr:>8.2e}  "
              f"({time.time()-t0:.1f}s){flag}")

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_f1": tr_f1,
            "val_loss": val_loss, "val_f1": val_f1, "val_auc": val_auc,
            "val_fpr": fpr, "lr": lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save({
                "epoch":        epoch,
                "model_type":   args.model,
                "model_state":  model.state_dict(),
                "val_loss":     val_loss,
                "val_f1":       val_f1,
                "val_auc":      val_auc,
                "input_size":   len(FEATURE_COLS),
                "hidden_size":  HIDDEN_SIZE,
                "num_layers":   NUM_LAYERS,
                "dropout":      DROPOUT,
            }, ckpt_path)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n[early stop] No val improvement for {PATIENCE} epochs.")
                break

    # ── Test evaluation ────────────────────────
    print("\n" + "─" * 78)
    print("[test] Loading best checkpoint for final evaluation...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_f1, test_auc, test_fpr = run_epoch(
        model, loaders["test"], optimizer, loss_fn, device, False
    )
    print(f"[test] F1={test_f1:.4f}  AUC={test_auc:.4f}  FPR={test_fpr:.4f}")
    print(f"\n[target] F1 > 0.70   {'PASS' if test_f1  > 0.70 else 'FAIL — need more data'}")
    print(f"[target] FPR < 0.15  {'PASS' if test_fpr < 0.15 else 'FAIL — adjust threshold'}")

    pd.DataFrame(history).to_csv(
        os.path.join(args.checkpoint_dir, "train_history.csv"), index=False
    )
    print(f"\n[saved] Best checkpoint → {ckpt_path}")
    print("[next]  Run: python model/export.py --checkpoint", ckpt_path)


# ─────────────────────────────────────────────
# Random Forest training
# ─────────────────────────────────────────────

def train_rf(args):
    norm_path  = os.path.join(os.path.dirname(args.data), "norm_stats.json")
    norm_stats = load_norm_stats(norm_path)
    df         = pd.read_csv(args.data)

    def get_split(split):
        subset = df[df["split"] == split]
        ds     = WindowSequenceDataset(subset, norm_stats)
        X      = ds.sequences    # (N, SEQ_LEN, 12)
        y      = ds.labels       # (N,)
        return X, y

    X_tr, y_tr = get_split("train")
    X_va, y_va = get_split("val")
    X_te, y_te = get_split("test")

    print(f"[rf] Train: {len(X_tr):,}  Val: {len(X_va):,}  Test: {len(X_te):,}")
    print(f"[rf] Confused in train: {int(y_tr.sum())} ({100*y_tr.mean():.1f}%)")

    rf = ConfusionRF(n_estimators=300, seed=args.seed)
    print("[rf] Training Random Forest (n_estimators=300, class_weight=balanced)...")
    t0 = time.time()
    rf.fit(X_tr, y_tr)
    print(f"[rf] Trained in {time.time()-t0:.1f}s")

    for split_name, X, y in [("val", X_va, y_va), ("test", X_te, y_te)]:
        preds = rf.predict(X)
        probs = rf.predict_proba(X)
        f1    = f1_score(y, preds,  zero_division=0)
        auc   = roc_auc_score(y, probs) if len(set(y)) > 1 else 0.5
        fpr   = _false_positive_rate(y.astype(int), preds.astype(int))
        print(f"[rf] {split_name:5s} — F1={f1:.4f}  AUC={auc:.4f}  FPR={fpr:.4f}")

    rf_path = os.path.join(args.checkpoint_dir, "rf_model.pkl")
    rf.save(rf_path)
    print(f"\n[saved] RF model → {rf_path}")
    print("[note]  RF cannot be exported to ONNX directly.")
    print("[note]  Once you have 500+ events, switch to: --model lstm")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Model: {args.model.upper()}   Device: {device}   Seed: {args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.model == "rf":
        train_rf(args)
    else:
        train_deep(args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default="lstm",
                        choices=["rf", "lstm", "transformer"])
    parser.add_argument("--data",           default="data/windows.csv")
    parser.add_argument("--pretrained",     default="model/checkpoints/pretrained_encoder.pt")
    parser.add_argument("--checkpoint-dir", default="model/checkpoints",
                        dest="checkpoint_dir")
    parser.add_argument("--epochs",         type=int, default=60)
    parser.add_argument("--seed",           type=int, default=42)
    main(parser.parse_args())