"""
pretrain.py
===========
Pre-trains a 2-layer LSTM encoder on the Kaggle Confused Student EEG dataset.

Goal: not to build a deployable EEG model, but to extract learned confusion
representations that initialise the LSTM before fine-tuning on keystroke data.
Only the encoder weights are saved — not the classification head.

Usage:
    python model/pretrain.py \
        --data data/eeg/EEG_data.csv \
        --out  model/checkpoints/pretrained_encoder.pt \
        --epochs 40 \
        --seed 42

Output:
    model/checkpoints/pretrained_encoder.pt   encoder weights only
    model/checkpoints/pretrain_stats.json     normalisation stats for EEG features
    model/checkpoints/pretrain_history.csv    loss and accuracy per epoch
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# 1.  Configuration
# ─────────────────────────────────────────────

EEG_FEATURE_COLS = [
    "Attention", "Mediation", "Raw",
    "Delta", "Theta",
    "Alpha1", "Alpha2",
    "Beta1", "Beta2",
    "Gamma1", "Gamma2",
]

LABEL_COL        = "user-definedlabeln"
SUBJECT_COL      = "SubjectID"
VIDEO_COL        = "VideoID"
SEQ_LEN          = 10       # windows per sequence fed to LSTM
HIDDEN_SIZE      = 64
NUM_LAYERS       = 2
DROPOUT          = 0.3
BATCH_SIZE       = 64
LEARNING_RATE    = 1e-3
POS_WEIGHT       = 2.0      # upweight confused class
PATIENCE         = 7        # early stopping


# ─────────────────────────────────────────────
# 2.  Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# 3.  Data loading and preprocessing
# ─────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load EEG CSV, validate expected columns, drop rows with nulls.
    The Kaggle dataset has 14 EEG band columns + a binary label column.
    """
    print(f"\n[data] Loading {csv_path}")
    df = pd.read_csv(csv_path)

    # Identify which feature columns are actually present
    present = [c for c in EEG_FEATURE_COLS if c in df.columns]
    missing = [c for c in EEG_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[warn] Missing EEG columns (will be skipped): {missing}")

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Label column '{LABEL_COL}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[present + [LABEL_COL, SUBJECT_COL, VIDEO_COL]].dropna()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    print(f"[data] Shape after cleaning : {df.shape}")
    print(f"[data] Label distribution   :")
    vc = df[LABEL_COL].value_counts()
    for label, count in vc.items():
        pct = 100 * count / len(df)
        print(f"         {label} ({'confused' if label==1 else 'not confused'}) "
              f"→ {count:,} rows ({pct:.1f}%)")

    return df, present


def build_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int):
    """
    Group rows by (SubjectID, VideoID), slide a window of seq_len rows,
    label the sequence with the majority label in the window.
    Returns X of shape (N, seq_len, n_features) and y of shape (N,).
    """
    X_list, y_list = [], []

    for (subject, video), group in df.groupby([SUBJECT_COL, VIDEO_COL]):
        group = group.reset_index(drop=True)
        vals  = group[feature_cols].values          # (T, F)
        labels = group[LABEL_COL].values            # (T,)

        if len(vals) < seq_len:
            continue

        for i in range(len(vals) - seq_len + 1):
            window   = vals[i : i + seq_len]        # (seq_len, F)
            majority = int(labels[i : i + seq_len].mean() >= 0.5)
            X_list.append(window)
            y_list.append(majority)

    X = np.array(X_list, dtype=np.float32)         # (N, seq_len, F)
    y = np.array(y_list, dtype=np.float32)          # (N,)
    print(f"[data] Sequences built: {X.shape}, "
          f"confused={int(y.sum())} ({100*y.mean():.1f}%)")
    return X, y


def split_by_subject(df, X, y, feature_cols, seq_len):
    """
    Subject-aware split: train on subjects 0-6, val on 7, test on 8-9.
    This prevents data leakage — same subject should not appear in
    both train and test sets.
    """
    subjects = df[SUBJECT_COL].values
    # Map each sequence back to its subject (first row of the window)
    seq_subjects = []
    for (subject, video), group in df.groupby([SUBJECT_COL, VIDEO_COL]):
        group = group.reset_index(drop=True)
        if len(group) < seq_len:
            continue
        for i in range(len(group) - seq_len + 1):
            seq_subjects.append(subject)
    seq_subjects = np.array(seq_subjects)

    train_mask = seq_subjects <= 6
    val_mask   = seq_subjects == 7
    test_mask  = seq_subjects >= 8

    return (
        X[train_mask], y[train_mask],
        X[val_mask],   y[val_mask],
        X[test_mask],  y[test_mask],
    )


# ─────────────────────────────────────────────
# 4.  PyTorch Dataset
# ─────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loader(X, y, batch_size, shuffle=True, oversample=False):
    dataset = EEGDataset(X, y)
    sampler = None

    if oversample and shuffle:
        # WeightedRandomSampler to balance classes in each batch
        class_counts = np.bincount(y.astype(int))
        weights      = 1.0 / class_counts[y.astype(int)]
        sampler      = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.float32),
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False     # sampler and shuffle are mutually exclusive

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )


# ─────────────────────────────────────────────
# 5.  Model definition
# ─────────────────────────────────────────────

class EEGEncoder(nn.Module):
    """
    2-layer LSTM encoder with attention pooling.
    The encoder maps (batch, seq_len, n_features) → (batch, hidden_size).
    Only this module's weights are saved to pretrained_encoder.pt.
    """
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        # Attention: learn which timestep matters most
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)                           # (batch, seq_len, H)
        attn_w = torch.softmax(self.attn(out), dim=1)   # (batch, seq_len, 1)
        context = (attn_w * out).sum(dim=1)             # (batch, H)
        return context, attn_w.squeeze(-1)


class EEGConfusionClassifier(nn.Module):
    """
    Full model = encoder + classification head.
    Only the encoder is saved after pre-training.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.encoder = EEGEncoder(input_size, hidden_size, num_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        context, attn_w = self.encoder(x)
        logits = self.head(context).squeeze(-1)     # (batch,)
        return logits, attn_w


# ─────────────────────────────────────────────
# 6.  Training utilities
# ─────────────────────────────────────────────

def run_epoch(model, loader, optimizer, loss_fn, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

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
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.long().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1       = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, f1


# ─────────────────────────────────────────────
# 7.  Main training loop
# ─────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[setup] Device: {device}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ── Load data ──────────────────────────────
    df, feature_cols = load_and_clean(args.data)
    X, y = build_sequences(df, feature_cols, SEQ_LEN)
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_by_subject(
        df, X, y, feature_cols, SEQ_LEN
    )

    print(f"\n[split] Train : {X_tr.shape[0]:,} sequences")
    print(f"[split] Val   : {X_val.shape[0]:,} sequences")
    print(f"[split] Test  : {X_te.shape[0]:,} sequences")

    # ── Normalise using train stats only ───────
    scaler = StandardScaler()
    n_feat = X_tr.shape[2]
    X_tr_2d  = X_tr.reshape(-1, n_feat)
    scaler.fit(X_tr_2d)
    X_tr  = scaler.transform(X_tr_2d ).reshape(X_tr.shape).astype(np.float32)
    X_val = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape).astype(np.float32)
    X_te  = scaler.transform(X_te.reshape(-1, n_feat) ).reshape(X_te.shape ).astype(np.float32)

    # Save normalisation stats
    stats = {
        "feature_cols" : feature_cols,
        "mean"         : scaler.mean_.tolist(),
        "std"          : scaler.scale_.tolist(),
    }
    stats_path = os.path.join(os.path.dirname(args.out), "pretrain_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[norm] Stats saved → {stats_path}")

    # ── DataLoaders ────────────────────────────
    train_loader = make_loader(X_tr,  y_tr,  BATCH_SIZE, shuffle=True,  oversample=True)
    val_loader   = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)
    test_loader  = make_loader(X_te,  y_te,  BATCH_SIZE, shuffle=False)

    # ── Model, loss, optimiser ─────────────────
    model = EEGConfusionClassifier(
        input_size  = len(feature_cols),
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
    ).to(device)

    pos_weight = torch.tensor([POS_WEIGHT]).to(device)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[model] Parameters: {total_params:,}")
    print(f"[model] Architecture:\n{model}\n")

    # ── Training loop ──────────────────────────
    best_val_loss  = float("inf")
    best_val_f1    = 0.0
    patience_count = 0
    history        = []

    print("─" * 65)
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train F1':>8}  "
          f"{'Val Loss':>8}  {'Val F1':>6}  {'LR':>8}")
    print("─" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1   = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss, val_f1 = run_epoch(model, val_loader,   optimizer, loss_fn, device, train=False)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_f1:>8.4f}  "
              f"{val_loss:>8.4f}  {val_f1:>6.4f}  {lr:>8.2e}  ({elapsed:.1f}s)")

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_f1": tr_f1,
            "val_loss": val_loss, "val_f1": val_f1, "lr": lr,
        })

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1   = val_f1
            patience_count = 0

            # Save encoder weights only (not the classification head)
            torch.save(
                {
                    "encoder_state_dict" : model.encoder.state_dict(),
                    "input_size"         : len(feature_cols),
                    "hidden_size"        : HIDDEN_SIZE,
                    "num_layers"         : NUM_LAYERS,
                    "dropout"            : DROPOUT,
                    "val_loss"           : val_loss,
                    "val_f1"             : val_f1,
                    "epoch"              : epoch,
                    "feature_cols"       : feature_cols,
                },
                args.out,
            )
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n[early stop] No improvement for {PATIENCE} epochs. Stopping.")
                break

    # ── Test evaluation ────────────────────────
    print("\n" + "─" * 65)
    print("[test] Evaluating on held-out test subjects (8 and 9)...")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits, _ = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_batch.long().numpy())

    test_f1  = f1_score(all_labels, all_preds, zero_division=0)
    test_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    print(f"[test] F1 score : {test_f1:.4f}")
    print(f"[test] AUC-ROC  : {test_auc:.4f}")
    print(f"\n[result] Best val loss : {best_val_loss:.4f}")
    print(f"[result] Best val F1   : {best_val_f1:.4f}")
    print(f"\n[saved] Encoder weights → {args.out}")
    print("[note]  These weights initialise the LSTM in train.py")
    print("[note]  Accuracy here does not matter much — the encoder")
    print("[note]  carries learned confusion representations forward.")

    # Save training history
    hist_path = os.path.join(os.path.dirname(args.out), "pretrain_history.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"[saved] Training history → {hist_path}")


# ─────────────────────────────────────────────
# 8.  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-train LSTM encoder on EEG confusion dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/eeg/EEG_data.csv",
        help="Path to the Kaggle EEG CSV file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="model/checkpoints/pretrained_encoder.pt",
        help="Where to save the encoder weights",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Maximum training epochs (early stopping may halt sooner)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    train(args)