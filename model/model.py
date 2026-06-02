"""
model.py
========
Model architectures for UnstuckAI confusion detection.

Three models, chosen based on how much labelled data you have:
  ConfusionRF          — Random Forest  (< 500 confusion events)
  ConfusionLSTM        — LSTM + attention (500–2000 events)  ← primary
  ConfusionTransformer — Transformer encoder (2000+ events)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from model.features import N_FEATURES


# ─────────────────────────────────────────────
# 1.  Shared encoder (used by LSTM and Transformer)
# ─────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """Weighted average pooling over a sequence using learned attention weights."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, H)
        w = torch.softmax(self.attn(x), dim=1)     # (batch, seq_len, 1)
        return (w * x).sum(dim=1), w.squeeze(-1)   # (batch, H), (batch, seq_len)


# ─────────────────────────────────────────────
# 2.  LSTM model — primary architecture
# ─────────────────────────────────────────────

class ConfusionLSTM(nn.Module):
    """
    2-layer LSTM with attention pooling and classification head.

    Input  : (batch, SEQ_LEN, N_FEATURES)
    Output : (batch,) logits — pass through sigmoid for probability
    """
    def __init__(
        self,
        input_size:  int   = N_FEATURES,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.attn = AttentionPooling(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _        = self.lstm(x)            # (batch, seq_len, H)
        context, attn = self.attn(out)          # (batch, H)
        logits        = self.head(context).squeeze(-1)
        return logits, attn

    def load_pretrained_encoder(self, checkpoint_path: str, device: str = "cpu"):
        """
        Load encoder weights from EEG pre-training checkpoint.
        Only loads weights that match in shape — safe to call even if
        the EEG encoder had a different input_size.
        """
        ckpt        = torch.load(checkpoint_path, map_location=device, weights_only=False)
        enc_state   = ckpt.get("encoder_state_dict", {})
        model_state = self.state_dict()

        loaded, skipped = [], []
        for name, param in enc_state.items():
            key = f"lstm.{name}" if not name.startswith("lstm") else name
            if key in model_state and model_state[key].shape == param.shape:
                model_state[key] = param
                loaded.append(key)
            else:
                skipped.append(name)

        self.load_state_dict(model_state)
        print(f"[pretrain] Loaded {len(loaded)} encoder layers, "
              f"skipped {len(skipped)} (shape mismatch or missing)")


# ─────────────────────────────────────────────
# 3.  Transformer model — upgrade path (2000+ events)
# ─────────────────────────────────────────────

class ConfusionTransformer(nn.Module):
    """
    Transformer encoder with 2 attention heads.
    Replace ConfusionLSTM with this when you have 2000+ confusion events.

    Input  : (batch, SEQ_LEN, N_FEATURES)
    Output : (batch,) logits
    """
    def __init__(
        self,
        input_size:  int = N_FEATURES,
        d_model:     int = 64,
        nhead:       int = 2,
        num_layers:  int = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn    = AttentionPooling(d_model)
        self.head    = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x             = self.input_proj(x)      # (batch, seq_len, d_model)
        out           = self.encoder(x)         # (batch, seq_len, d_model)
        context, attn = self.attn(out)
        logits        = self.head(context).squeeze(-1)
        return logits, attn


# ─────────────────────────────────────────────
# 4.  Random Forest wrapper — first baseline
# ─────────────────────────────────────────────

class ConfusionRF:
    """
    Scikit-learn Random Forest wrapper that accepts the same
    (N, SEQ_LEN, 12) input as the PyTorch models by flattening
    the sequence dimension and adding lag statistics.
    """
    def __init__(self, n_estimators: int = 200, seed: int = 42):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators = n_estimators,
                class_weight = "balanced",
                max_depth    = 12,
                random_state = seed,
                n_jobs       = -1,
            )),
        ])

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        # X: (N, SEQ_LEN, 12)
        # Flatten to (N, SEQ_LEN*12) and add summary stats
        N, T, F = X.shape
        flat    = X.reshape(N, T * F)
        mean    = X.mean(axis=1)        # (N, F)
        std     = X.std(axis=1)         # (N, F)
        trend   = X[:, -1, :] - X[:, 0, :]   # (N, F) — change from first to last window
        return np.concatenate([flat, mean, std, trend], axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(self._flatten(X), y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._flatten(X))[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._flatten(X))

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[model] RF saved → {path}")

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            self.model = pickle.load(f)


# ─────────────────────────────────────────────
# 5.  Factory
# ─────────────────────────────────────────────

def build_model(model_type: str, **kwargs):
    if model_type == "rf":
        return ConfusionRF(**kwargs)
    elif model_type == "lstm":
        return ConfusionLSTM(**kwargs)
    elif model_type == "transformer":
        return ConfusionTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose rf, lstm, or transformer.")