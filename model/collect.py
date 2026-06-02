"""
collect.py
==========
Reads raw session JSONL logs written by the VS Code extension,
aligns self-report confusion labels with a look-back window,
extracts rolling-window features, and outputs a training-ready CSV.

Usage:
    python model/collect.py \
        --input  data/sessions.jsonl \
        --output data/windows.csv \
        --lookback 30 \
        --min-session-minutes 10

Output:
    data/windows.csv        feature matrix with labels, one row per window
    data/norm_stats.json    normalisation stats (mean/std per feature)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.features import (
    extract_windows, compute_norm_stats,
    save_norm_stats, FEATURE_COLS, generate_synthetic_session
)


# ─────────────────────────────────────────────
# 1.  Load and parse JSONL
# ─────────────────────────────────────────────

def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        raise ValueError(f"No valid records found in {path}")

    df = pd.DataFrame(records)
    print(f"[load] Loaded {len(df):,} raw events from {path}")
    print(f"[load] Sessions found: {df['session_id'].nunique()}")
    print(f"[load] Date range: {pd.to_datetime(df['timestamp'], unit='s').min().date()} "
          f"→ {pd.to_datetime(df['timestamp'], unit='s').max().date()}")
    return df


# ─────────────────────────────────────────────
# 2.  Label alignment
# ─────────────────────────────────────────────

def align_labels(df: pd.DataFrame, lookback_sec: float = 30.0) -> pd.DataFrame:
    """
    Self-report labels are pressed AFTER confusion onset, not during it.
    This function back-propagates the label to events within lookback_sec
    before the shortcut press, creating a more accurate training signal.
    """
    df = df.copy().sort_values(["session_id", "timestamp"]).reset_index(drop=True)

    # Find all rows where confused=1 (shortcut was pressed)
    label_rows = df[df["confused"] == 1].copy()

    if label_rows.empty:
        print("[labels] No self-report labels found. "
              "All windows will be labelled 0 (not confused).")
        return df

    print(f"[labels] Found {len(label_rows):,} confusion shortcut presses")

    # For each label event, mark all events within lookback_sec before it
    for _, label_row in label_rows.iterrows():
        session = label_row["session_id"]
        t_label = label_row["timestamp"]
        t_start = t_label - lookback_sec

        mask = (
            (df["session_id"] == session) &
            (df["timestamp"]  >= t_start) &
            (df["timestamp"]  <= t_label) &
            (df["confused"]   == 0)
        )
        df.loc[mask, "confused"] = 1

    n_confused = (df["confused"] == 1).sum()
    print(f"[labels] After look-back alignment: {n_confused:,} confused events "
          f"({100*n_confused/len(df):.1f}%)")
    return df


# ─────────────────────────────────────────────
# 3.  Session filtering
# ─────────────────────────────────────────────

def filter_sessions(df: pd.DataFrame, min_minutes: float = 10.0) -> pd.DataFrame:
    kept = []
    for session_id, group in df.groupby("session_id"):
        duration_min = (group["timestamp"].max() - group["timestamp"].min()) / 60.0
        if duration_min >= min_minutes:
            kept.append(group)
        else:
            print(f"[filter] Dropped session {session_id} "
                  f"({duration_min:.1f} min — below {min_minutes} min threshold)")

    if not kept:
        raise ValueError("No sessions passed the minimum duration filter.")

    result = pd.concat(kept, ignore_index=True)
    print(f"[filter] Kept {result['session_id'].nunique()} sessions, "
          f"{len(result):,} events")
    return result


# ─────────────────────────────────────────────
# 4.  Feature extraction across all sessions
# ─────────────────────────────────────────────

def extract_all_sessions(df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []

    for session_id, group in df.groupby("session_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        X, y, ts = extract_windows(group)

        if len(X) == 0:
            print(f"[extract] Session {session_id}: no windows extracted (too short)")
            continue

        session_df = pd.DataFrame(X, columns=FEATURE_COLS)
        session_df["confused"]   = y.astype(int)
        session_df["timestamp"]  = ts
        session_df["session_id"] = session_id
        all_rows.append(session_df)

        confused_pct = 100 * y.mean()
        print(f"[extract] Session {session_id}: "
              f"{len(X):,} windows, {confused_pct:.1f}% confused")

    if not all_rows:
        raise ValueError("No windows could be extracted from any session.")

    result = pd.concat(all_rows, ignore_index=True)
    print(f"\n[extract] Total windows : {len(result):,}")
    print(f"[extract] Confused      : {result['confused'].sum():,} "
          f"({100*result['confused'].mean():.1f}%)")
    print(f"[extract] Not confused  : {(result['confused']==0).sum():,}")
    return result


# ─────────────────────────────────────────────
# 5.  Session-aware train/val/test split
# ─────────────────────────────────────────────

def split_by_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split chronologically by session — never randomly.
    Adds a 'split' column: train / val / test.
    """
    sessions = df.groupby("session_id")["timestamp"].min().sort_values()
    n = len(sessions)

    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train_sessions = sessions.index[:train_end].tolist()
    val_sessions   = sessions.index[train_end:val_end].tolist()
    test_sessions  = sessions.index[val_end:].tolist()

    df = df.copy()
    df["split"] = df["session_id"].map(
        {s: "train" for s in train_sessions} |
        {s: "val"   for s in val_sessions}   |
        {s: "test"  for s in test_sessions}
    )

    for split in ["train", "val", "test"]:
        subset = df[df["split"] == split]
        print(f"[split] {split:5s} — {len(subset):,} windows, "
              f"{subset['session_id'].nunique()} sessions, "
              f"{100*subset['confused'].mean():.1f}% confused")

    return df


# ─────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────

def main(args):
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ── Load or generate synthetic data ────────
    if not os.path.exists(args.input):
        print(f"[warn] {args.input} not found.")
        print("[warn] Generating synthetic session data for demonstration...")

        from model.features import generate_synthetic_session
        sessions = []
        for i in range(12):
            df_s = generate_synthetic_session(
                n_events   = 800,
                n_confused = 60,
                seed       = i,
            )
            df_s["session_id"] = f"synthetic_{i:03d}"
            sessions.append(df_s)

        df_raw = pd.concat(sessions, ignore_index=True)

        os.makedirs(os.path.dirname(args.input) or ".", exist_ok=True)
        jsonl_path = args.input
        with open(jsonl_path, "w") as f:
            for _, row in df_raw.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")
        print(f"[warn] Synthetic JSONL written → {jsonl_path}")
        print("[warn] Replace this with your real session logs.\n")

    # ── Full pipeline ──────────────────────────
    df_raw = load_jsonl(args.input)

    # Ensure required columns
    for col in ["timestamp", "session_id", "confused"]:
        if col not in df_raw.columns:
            raise ValueError(f"Required column '{col}' missing from JSONL.")

    # Fill optional columns with defaults
    for col, default in [
        ("is_delete", 0), ("idle_sec", 0.1),
        ("error_count", 0), ("error_type", None),
        ("line_number", 1),
    ]:
        if col not in df_raw.columns:
            df_raw[col] = default

    df_aligned  = align_labels(df_raw, lookback_sec=args.lookback)
    df_filtered = filter_sessions(df_aligned, min_minutes=args.min_session_minutes)
    df_windows  = extract_all_sessions(df_filtered)
    df_split    = split_by_session(df_windows)

    # ── Save windows CSV ───────────────────────
    df_split.to_csv(args.output, index=False)
    print(f"\n[saved] Windows CSV → {args.output}")

    # ── Compute and save norm stats (train only) ─
    train_X = df_split[df_split["split"] == "train"][FEATURE_COLS].values
    stats   = compute_norm_stats(train_X)
    stats_path = os.path.join(os.path.dirname(args.output), "norm_stats.json")
    save_norm_stats(stats, stats_path)

    # ── Summary ───────────────────────────────
    print("\n" + "─" * 50)
    print("Dataset summary")
    print("─" * 50)
    print(f"  Total windows    : {len(df_split):,}")
    print(f"  Features         : {len(FEATURE_COLS)}")
    print(f"  Confused windows : {df_split['confused'].sum():,} "
          f"({100*df_split['confused'].mean():.1f}%)")
    print(f"  Sessions used    : {df_split['session_id'].nunique()}")
    print(f"\n  Output CSV       : {args.output}")
    print(f"  Norm stats       : {stats_path}")
    print("\nReady for training. Run:")
    print("  python model/train.py --model rf   --data", args.output, "(< 500 confused events)")
    print("  python model/train.py --model lstm --data", args.output, "(500+ confused events)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training dataset from session logs")
    parser.add_argument("--input",               default="data/sessions.jsonl")
    parser.add_argument("--output",              default="data/windows.csv")
    parser.add_argument("--lookback",            type=float, default=30.0)
    parser.add_argument("--min-session-minutes", type=float, default=10.0,
                        dest="min_session_minutes")
    main(parser.parse_args())