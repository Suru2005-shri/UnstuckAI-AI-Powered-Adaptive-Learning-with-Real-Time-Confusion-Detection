"""
dataset.py
==========
PyTorch Dataset and DataLoader factory for UnstuckAI keystroke windows.
Packs windows into sequences of SEQ_LEN for LSTM input.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from model.features import FEATURE_COLS, apply_norm, load_norm_stats

SEQ_LEN = 10    # how many consecutive windows feed into the LSTM


class WindowSequenceDataset(Dataset):
    """
    Packs consecutive feature windows into sequences of length SEQ_LEN.
    Sequences are built per-session so no session boundary is crossed.
    Label = majority label across the sequence.
    """
    def __init__(self, df: pd.DataFrame, norm_stats: dict):
        self.sequences = []
        self.labels    = []

        for session_id, group in df.groupby("session_id"):
            group  = group.sort_values("timestamp").reset_index(drop=True)
            X      = group[FEATURE_COLS].values.astype(np.float32)
            y      = group["confused"].values.astype(np.float32)
            X_norm = apply_norm(X, norm_stats)

            for i in range(len(X_norm) - SEQ_LEN + 1):
                seq   = X_norm[i : i + SEQ_LEN]       # (SEQ_LEN, 12)
                label = float(y[i : i + SEQ_LEN].mean() >= 0.5)
                self.sequences.append(seq)
                self.labels.append(label)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels    = np.array(self.labels,    dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],    dtype=torch.float32),
        )


def make_loaders(windows_csv: str, norm_stats_path: str, batch_size: int = 64):
    df         = pd.read_csv(windows_csv)
    norm_stats = load_norm_stats(norm_stats_path)

    splits  = {}
    loaders = {}
    for split in ["train", "val", "test"]:
        subset = df[df["split"] == split]
        if subset.empty:
            continue
        splits[split] = WindowSequenceDataset(subset, norm_stats)

    # Training loader uses WeightedRandomSampler to balance classes
    y_train = splits["train"].labels
    class_counts = np.bincount(y_train.astype(int))
    sample_weights = 1.0 / class_counts[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float32),
        num_samples = len(sample_weights),
        replacement = True,
    )
    loaders["train"] = DataLoader(
        splits["train"], batch_size=batch_size, sampler=sampler, num_workers=0
    )
    for split in ["val", "test"]:
        if split in splits:
            loaders[split] = DataLoader(
                splits[split], batch_size=batch_size, shuffle=False, num_workers=0
            )

    return loaders, splits