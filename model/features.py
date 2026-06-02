"""
features.py
===========
Computes 20 behavioural features from raw session event logs.

THIS FILE IS THE SINGLE SOURCE OF TRUTH for feature definitions.
extension/src/features.ts must mirror every computation here exactly.

Features computed per 30-second rolling window (5-second step):

  Original 12 signals:
  1.  wpm_mean             average typing speed (words per minute)
  2.  wpm_std              variability in typing speed
  3.  wpm_trend            slope: is speed falling or recovering?
  4.  delete_ratio_mean    average fraction of keystrokes that are deletions
  5.  delete_ratio_max     worst-case deletion spike in the window
  6.  idle_mean            average pause duration between keystrokes (seconds)
  7.  idle_max             longest single pause in the window
  8.  idle_count           number of pauses longer than 3 seconds
  9.  error_count          total diagnostics in Problems panel
  10. error_repeat_flag    1 if same error type appeared 3+ times
  11. time_on_line_sec     seconds cursor spent on the same line
  12. cursor_moves         number of distinct line changes

  New rich context signals:
  13. attempt_count        how many times user tried to fix the same problem
  14. same_error_count     how many times the exact same error appeared
  15. run_count            how many times code was executed in this window
  16. hint_clicked         1 if user clicked a hint or VS Code suggestion
  17. question_asked       1 if user opened browser/search during this window
  18. cartoon_activated    1 if UnstuckAI character appeared this window
  19. understood_after_help 1 if confusion score dropped after explanation
  20. confusion_type_id    numeric ID of detected confusion type (C01-C27)
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple


# ─────────────────────────────────────────────
# Constants — must match logger.ts exactly
# ─────────────────────────────────────────────

WINDOW_SEC  = 30
STEP_SEC    = 5
IDLE_THRESH = 3.0

FEATURE_COLS = [
    # Original behavioural signals
    "wpm_mean", "wpm_std", "wpm_trend",
    "delete_ratio_mean", "delete_ratio_max",
    "idle_mean", "idle_max", "idle_count",
    "error_count", "error_repeat_flag",
    "time_on_line_sec", "cursor_moves",
    # New rich context signals
    "attempt_count",
    "same_error_count",
    "run_count",
    "hint_clicked",
    "question_asked",
    "cartoon_activated",
    "understood_after_help",
    "confusion_type_id",
]
N_FEATURES = len(FEATURE_COLS)   # 20

# Confusion type ID map — C01 to C27
CONFUSION_TYPE_MAP = {
    "syntax_error":          1,
    "name_error":            2,
    "indentation":           3,
    "blank_line":            4,
    "type_error":            5,
    "loop_boundary":         6,
    "wrong_return":          7,
    "scope":                 8,
    "list_mutation":         9,
    "index_error":          10,
    "mutable_default":      11,
    "reference_copy":       12,
    "key_error":            13,
    "recursion":            14,
    "class_self":           15,
    "async_await":          16,
    "decorator":            17,
    "generator":            18,
    "import_error":         19,
    "threading":            20,
    "memory_leak":          21,
    "api_response":         22,
    "metaclass":            23,
    "gil":                  24,
    "ml_convergence":       25,
    "cuda_device":          26,
    "train_serve_skew":     27,
    "unknown":               0,
}


# ─────────────────────────────────────────────
# Core feature computation
# ─────────────────────────────────────────────

def compute_features(window: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame of raw events within a 30-second window,
    return a numpy array of shape (20,) with all features.

    Expected columns in window:
        timestamp           float   Unix timestamp
        is_delete           int     1 = deletion
        idle_sec            float   seconds since previous keystroke
        error_count         int     Problems panel count
        error_type          str     e.g. NameError
        line_number         int     cursor line
        attempt_count       int     fix attempts on same error
        same_error_count    int     repeated identical error count
        run_count           int     code executions in window
        hint_clicked        int     1 if hint used
        question_asked      int     1 if searched externally
        cartoon_activated   int     1 if character appeared
        understood_after_help int   1 if confusion resolved after help
        confusion_type      str     one of CONFUSION_TYPE_MAP keys
    """
    feats = np.zeros(N_FEATURES, dtype=np.float32)

    if window.empty or len(window) < 2:
        return feats

    # ── 1-3: Typing speed (WPM) ──────────────
    duration_min = max(
        (window["timestamp"].max() - window["timestamp"].min()) / 60.0, 1e-6
    )
    wpm_series = window.apply(
        lambda r: (1 / 5) / max(r["idle_sec"], 0.01) * 60
        if r["is_delete"] == 0 else 0, axis=1
    ).replace([np.inf, -np.inf], 0).clip(0, 300)

    feats[0] = float(wpm_series.mean())
    feats[1] = float(wpm_series.std()) if len(wpm_series) > 1 else 0.0
    if len(wpm_series) >= 3:
        t = np.arange(len(wpm_series), dtype=np.float32)
        feats[2] = float(np.clip(np.polyfit(t, wpm_series.values, 1)[0], -50, 50))

    # ── 4-5: Delete ratio ─────────────────────
    total_keys = len(window)
    chunk_size = max(5, total_keys // 6)
    ratios = [
        window.iloc[i: i + chunk_size]["is_delete"].mean()
        for i in range(0, total_keys, chunk_size)
        if len(window.iloc[i: i + chunk_size]) > 0
    ]
    feats[3] = float(np.mean(ratios)) if ratios else 0.0
    feats[4] = float(np.max(ratios))  if ratios else 0.0

    # ── 6-8: Idle pauses ──────────────────────
    idles    = window["idle_sec"].clip(0, 120).values
    feats[5] = float(idles.mean())
    feats[6] = float(idles.max())
    feats[7] = float((idles > IDLE_THRESH).sum())

    # ── 9-10: Errors ──────────────────────────
    feats[8] = float(window["error_count"].max())
    if "error_type" in window.columns:
        ec       = window["error_type"].dropna().value_counts()
        feats[9] = 1.0 if (ec >= 3).any() else 0.0

    # ── 11-12: Cursor behaviour ───────────────
    if "line_number" in window.columns:
        lines = window["line_number"].values
        from collections import Counter
        most_visited = max(Counter(lines).values())
        feats[10] = float(most_visited / len(lines) * duration_min * 60)
        feats[11] = float(len(np.where(np.diff(lines) != 0)[0]))

    # ── 13: attempt_count ─────────────────────
    if "attempt_count" in window.columns:
        feats[12] = float(window["attempt_count"].max())

    # ── 14: same_error_count ──────────────────
    if "same_error_count" in window.columns:
        feats[13] = float(window["same_error_count"].max())

    # ── 15: run_count ─────────────────────────
    if "run_count" in window.columns:
        feats[14] = float(window["run_count"].sum())

    # ── 16: hint_clicked ──────────────────────
    if "hint_clicked" in window.columns:
        feats[15] = 1.0 if window["hint_clicked"].max() >= 1 else 0.0

    # ── 17: question_asked ────────────────────
    if "question_asked" in window.columns:
        feats[16] = 1.0 if window["question_asked"].max() >= 1 else 0.0

    # ── 18: cartoon_activated ─────────────────
    if "cartoon_activated" in window.columns:
        feats[17] = 1.0 if window["cartoon_activated"].max() >= 1 else 0.0

    # ── 19: understood_after_help ────────────
    if "understood_after_help" in window.columns:
        feats[18] = 1.0 if window["understood_after_help"].max() >= 1 else 0.0

    # ── 20: confusion_type_id ────────────────
    if "confusion_type" in window.columns:
        ct = window["confusion_type"].dropna()
        if not ct.empty:
            most_common_type = ct.mode()[0] if not ct.mode().empty else "unknown"
            feats[19] = float(CONFUSION_TYPE_MAP.get(most_common_type, 0))

    return feats


# ─────────────────────────────────────────────
# Window sliding over a session
# ─────────────────────────────────────────────

def extract_windows(
    session_df: pd.DataFrame,
    window_sec: float = WINDOW_SEC,
    step_sec:   float = STEP_SEC,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if session_df.empty:
        return np.zeros((0, N_FEATURES)), np.zeros(0), np.zeros(0)

    t_min = session_df["timestamp"].min()
    t_max = session_df["timestamp"].max()
    X_list, y_list, ts_list = [], [], []

    t = t_min + window_sec
    while t <= t_max + step_sec:
        window = session_df[
            (session_df["timestamp"] >= t - window_sec) &
            (session_df["timestamp"] <  t)
        ]
        if len(window) >= 3:
            feats = compute_features(window)
            label = int(window["confused"].max()) if "confused" in window.columns else 0
            X_list.append(feats)
            y_list.append(label)
            ts_list.append(t)
        t += step_sec

    if not X_list:
        return np.zeros((0, N_FEATURES)), np.zeros(0), np.zeros(0)

    return (
        np.array(X_list,  dtype=np.float32),
        np.array(y_list,  dtype=np.float32),
        np.array(ts_list, dtype=np.float64),
    )


# ─────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────

def compute_norm_stats(X: np.ndarray) -> Dict:
    mean = X.mean(axis=0).tolist()
    std  = [max(s, 1e-6) for s in X.std(axis=0).tolist()]
    return {"feature_cols": FEATURE_COLS, "mean": mean, "std": std}


def save_norm_stats(stats: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[norm] Stats saved → {path}")


def load_norm_stats(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def apply_norm(X: np.ndarray, stats: Dict) -> np.ndarray:
    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.array(stats["std"],  dtype=np.float32)
    return (X - mean) / std


# ─────────────────────────────────────────────
# Synthetic session generator (for testing)
# ─────────────────────────────────────────────

def generate_synthetic_session(
    n_events:   int = 500,
    n_confused: int = 40,
    seed:       int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_ts = 1_716_720_000.0
    events  = []
    confused_timestamps = set(
        rng.choice(range(60, n_events - 30), size=n_confused, replace=False).tolist()
    )

    ts           = base_ts
    current_line = 1
    attempt_cnt  = 0
    same_err_cnt = 0
    run_cnt      = 0
    last_error   = None

    for i in range(n_events):
        in_confused = any(abs(i - c) < 15 for c in confused_timestamps)
        idle        = float(rng.exponential(0.4 if not in_confused else 2.5))
        idle        = min(idle, 60.0)
        ts         += idle
        is_delete   = int(rng.random() < (0.08 if not in_confused else 0.45))

        if rng.random() < 0.05:
            current_line = max(1, current_line + int(rng.integers(-3, 4)))

        error_count = 0
        error_type  = None
        if in_confused and rng.random() < 0.4:
            error_count = int(rng.integers(1, 5))
            error_type  = rng.choice(["NameError", "TypeError", "IndentationError"])
            if error_type == last_error:
                same_err_cnt += 1
                attempt_cnt  += 1
            else:
                same_err_cnt = 1
                attempt_cnt  = 1
            last_error = error_type

        if rng.random() < 0.03:
            run_cnt += 1

        confused = 1 if (i in confused_timestamps and rng.random() < 0.6) else 0

        # New fields
        hint_clicked        = int(in_confused and rng.random() < 0.15)
        question_asked      = int(in_confused and rng.random() < 0.20)
        cartoon_activated   = int(in_confused and rng.random() < 0.10)
        understood_after    = int(cartoon_activated and rng.random() < 0.60)
        confusion_type      = rng.choice(list(CONFUSION_TYPE_MAP.keys())) if in_confused else "unknown"

        events.append({
            "timestamp":            ts,
            "is_delete":            is_delete,
            "idle_sec":             idle,
            "error_count":          error_count,
            "error_type":           error_type,
            "line_number":          current_line,
            "confused":             confused,
            "session_id":           f"synthetic_{seed:03d}",
            "attempt_count":        attempt_cnt,
            "same_error_count":     same_err_cnt,
            "run_count":            run_cnt,
            "hint_clicked":         hint_clicked,
            "question_asked":       question_asked,
            "cartoon_activated":    cartoon_activated,
            "understood_after_help": understood_after,
            "confusion_type":       confusion_type,
        })

    return pd.DataFrame(events)


# ─────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("[features.py] Running self-test with 20-feature schema...")
    df = generate_synthetic_session(n_events=600, n_confused=50, seed=42)
    print(f"  Session rows      : {len(df)}")
    print(f"  Columns           : {list(df.columns)}")

    X, y, ts = extract_windows(df)
    print(f"\n  Windows extracted : {X.shape[0]}")
    print(f"  Feature shape     : {X.shape}  (expected N x {N_FEATURES})")
    print(f"  Confused windows  : {int(y.sum())} ({100*y.mean():.1f}%)")
    print(f"\n  Feature names     :")
    for i, name in enumerate(FEATURE_COLS):
        print(f"    {i+1:2d}. {name:<30} mean={X[:,i].mean():.3f}")

    stats  = compute_norm_stats(X)
    X_norm = apply_norm(X, stats)
    print(f"\n  Normalised mean   : {X_norm.mean(axis=0).round(2)}")
    print(f"\n[features.py] Self-test passed. N_FEATURES = {N_FEATURES}")