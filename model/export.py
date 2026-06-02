"""
export.py
=========
Exports the trained PyTorch confusion model to ONNX format
and copies it into the VS Code extension's models/ directory.

Usage:
    python model/export.py \
        --checkpoint model/checkpoints/best.pt \
        --output     extension/models/confusion_model.onnx
"""

import argparse
import os
import sys
import json
import shutil

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.model   import build_model
from model.features import FEATURE_COLS, N_FEATURES
from model.dataset  import SEQ_LEN


def export(args):
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    device = torch.device("cpu")   # ONNX export always on CPU
    print(f"\n[export] Loading checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "lstm")
    print(f"[export] Model type : {model_type.upper()}")
    print(f"[export] Trained at epoch {ckpt.get('epoch', '?')} "
          f"— val_f1={ckpt.get('val_f1', 0):.4f}")

    # ── Rebuild model and load weights ─────────
    model = build_model(
        model_type,
        input_size  = ckpt.get("input_size",  N_FEATURES),
        hidden_size = ckpt.get("hidden_size", 64),
        num_layers  = ckpt.get("num_layers",  2),
        dropout     = ckpt.get("dropout",     0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Dummy input for tracing ─────────────────
    # Shape: (batch=1, seq_len=10, n_features=12)
    dummy_input = torch.zeros(1, SEQ_LEN, N_FEATURES, dtype=torch.float32)

    # ── Export to ONNX ─────────────────────────
    print(f"[export] Exporting to ONNX → {args.output}")
    torch.onnx.export(
        model,
        (dummy_input,),
        args.output,
        input_names    = ["windows"],
        output_names   = ["logits", "attention"],
        dynamic_axes   = {
            "windows":   {0: "batch"},
            "logits":    {0: "batch"},
            "attention": {0: "batch"},
        },
        opset_version  = 17,
        export_params  = True,
        do_constant_folding = True,
    )
    print(f"[export] ONNX file size: {os.path.getsize(args.output) / 1024:.1f} KB")

    # ── Validate ONNX output matches PyTorch ───
    print("[export] Validating ONNX output vs PyTorch...")
    import onnxruntime as ort

    ort_session = ort.InferenceSession(args.output)
    test_input  = np.random.randn(1, SEQ_LEN, N_FEATURES).astype(np.float32)

    with torch.no_grad():
        pt_logits, _ = model(torch.tensor(test_input))
        pt_prob      = torch.sigmoid(pt_logits).item()

    ort_outputs  = ort_session.run(None, {"windows": test_input})
    ort_prob     = float(1 / (1 + np.exp(-ort_outputs[0][0])))   # sigmoid

    diff = abs(pt_prob - ort_prob)
    print(f"[export] PyTorch prob : {pt_prob:.6f}")
    print(f"[export] ONNX prob    : {ort_prob:.6f}")
    print(f"[export] Difference   : {diff:.2e}  {'PASS' if diff < 1e-4 else 'WARN — large diff'}")

    # ── Copy norm_stats.json alongside ONNX ────
    norm_src = os.path.join(os.path.dirname(args.checkpoint).replace("checkpoints", ""),
                            "..", "data", "norm_stats.json")
    norm_src = os.path.normpath(norm_src)
    norm_dst = os.path.join(os.path.dirname(args.output), "norm_stats.json")

    if os.path.exists(norm_src):
        shutil.copy(norm_src, norm_dst)
        print(f"[export] norm_stats.json → {norm_dst}")
    else:
        print(f"[warn]  norm_stats.json not found at {norm_src}")
        print(f"[warn]  Copy data/norm_stats.json to {os.path.dirname(args.output)}/ manually")

    # ── Write model metadata ────────────────────
    metadata = {
        "model_type":   model_type,
        "input_shape":  [1, SEQ_LEN, N_FEATURES],
        "output":       "logit — apply sigmoid for confusion probability",
        "threshold":    0.65,
        "feature_cols": FEATURE_COLS,
        "val_f1":       ckpt.get("val_f1"),
        "val_auc":      ckpt.get("val_auc"),
        "epoch":        ckpt.get("epoch"),
    }
    meta_path = os.path.join(os.path.dirname(args.output), "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[export] Metadata → {meta_path}")

    # ── Final summary ──────────────────────────
    print("\n" + "─" * 55)
    print("Export complete")
    print("─" * 55)
    print(f"  ONNX model   : {args.output}")
    print(f"  Norm stats   : {norm_dst}")
    print(f"  Metadata     : {meta_path}")
    print(f"  Input shape  : [1, {SEQ_LEN}, {N_FEATURES}]")
    print(f"  Threshold    : 0.65 (Level 2 activation)")
    print("\nIn VS Code extension (scorer.ts):")
    print("  const session = await InferenceSession.create('confusion_model.onnx')")
    print("  const result  = await session.run({ windows: featureTensor })")
    print("  const score   = sigmoid(result.logits.data[0])")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model/checkpoints/best.pt")
    parser.add_argument("--output",     default="extension/models/confusion_model.onnx")
    export(parser.parse_args())