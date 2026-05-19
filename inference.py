"""
inference.py
------------
5-Fold Softmax Averaging Ensemble for the HAR 1D-ResNet (v2).

Strategy
────────
  For each sample in the held-out test set:
    1. Load all five fold checkpoints  (fold_1_best.pth … fold_5_best.pth).
    2. Run each model in eval mode to obtain raw logits of shape (B, 6).
    3. Convert logits to probability distributions via Softmax.
    4. Average the 5 probability matrices element-wise.
    5. Take argmax of the averaged probability vector as the final prediction.

  This soft-probability aggregation reduces variance compared to hard-voting
  and eliminates Private LB shake-up caused by individual fold outliers.

Usage
─────
    conda activate PyTorch
    python inference.py                        # uses checkpoints in ./checkpoints/
    python inference.py --ckpt-dir /path/to/   # custom checkpoint directory
    python inference.py --output my_sub.csv    # custom output path

Output
──────
    submission.csv  (or custom path)  with columns:  Id, Label
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (
    load_all_samples, get_fold_splits,
    compute_normalization_params, normalize,
    load_test_samples, HARTestDataset,
    N_FOLDS, N_CLASSES
)
from model import ResNet1D

# ──────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────
DEFAULT_CFG = {
    "train_root"     : "./train/train",
    "test_root"      : "./test/test",
    "checkpoint_dir" : "./checkpoints",
    "output_path"    : "./submission.csv",
    "batch_size"     : 256,
    "num_workers"    : 4,
    "pin_memory"     : True,
    "in_channels"    : 6,
    "num_classes"    : 6,
    "base_filters"   : 64,
}


# ══════════════════════════════════════════════════════════════════════════════
# Helper: load a single model checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, cfg: dict,
               device: torch.device) -> ResNet1D:
    """Instantiate ResNet1D and load weights from *checkpoint_path*."""
    model = ResNet1D(
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
        base_filters=cfg["base_filters"],
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Helper: compute per-fold normalisation parameters
# ══════════════════════════════════════════════════════════════════════════════

def get_fold_norm_params(train_root: str, fold_idx: int):
    """
    Reproduce the exact same normalisation parameters that were used when
    training fold *fold_idx*.  Uses the same StratifiedGroupKFold split with
    random_state=42 so results are deterministic.

    Returns
    -------
    mean : np.ndarray, shape (6,)
    std  : np.ndarray, shape (6,)
    """
    signals, labels, groups, _ = load_all_samples(train_root)
    train_idx, _               = get_fold_splits(signals, labels, groups, fold_idx)
    mean, std                  = compute_normalization_params(signals[train_idx])
    return mean, std


# ══════════════════════════════════════════════════════════════════════════════
# Core ensemble inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def ensemble_predict(cfg: dict, device: torch.device,
                     folds: list) -> pd.DataFrame:
    """
    Run softmax averaging ensemble over *folds* checkpoints.

    Parameters
    ----------
    cfg    : configuration dict
    device : torch.device
    folds  : list of 1-indexed fold numbers to include (e.g. [1,2,3,4,5])

    Returns
    -------
    pd.DataFrame with columns  Id | Label,  sorted by Id
    """
    raw_signals, file_ids = load_test_samples(cfg["test_root"])
    n_samples = len(file_ids)

    # Accumulator for softmax probabilities  (N, C)
    prob_sum = np.zeros((n_samples, N_CLASSES), dtype=np.float64)

    for fold_num in folds:
        ckpt_path = os.path.join(cfg["checkpoint_dir"],
                                 f"fold_{fold_num}_best.pth")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Run train.py --all-folds first."
            )

        # Reproduce the normalisation parameters from this fold's training split
        fold_idx  = fold_num - 1   # convert to 0-indexed
        mean, std = get_fold_norm_params(cfg["train_root"], fold_idx)

        # Build test dataset with this fold's normalisation
        test_signals = normalize(raw_signals, mean, std)
        test_ds      = HARTestDataset(test_signals, file_ids)
        test_loader  = DataLoader(
            test_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=cfg["pin_memory"],
        )

        model = load_model(ckpt_path, cfg, device)
        print(f"  Loaded fold {fold_num} checkpoint: {ckpt_path}")

        # Collect softmax probabilities for the entire test set
        fold_probs = []
        for signals, _ in test_loader:
            logits = model(signals.to(device, non_blocking=True))  # (B, C)
            probs  = F.softmax(logits, dim=1).cpu().numpy()        # (B, C)
            fold_probs.append(probs)

        fold_probs = np.concatenate(fold_probs, axis=0)   # (N, C)
        prob_sum  += fold_probs

    # Average across folds and take argmax
    prob_avg   = prob_sum / len(folds)          # (N, C)
    preds      = prob_avg.argmax(axis=1)        # (N,)

    df = (pd.DataFrame({"Id": file_ids, "Label": preds})
          .sort_values("Id")
          .reset_index(drop=True))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HAR 1D-ResNet 5-Fold Softmax Averaging Ensemble"
    )
    parser.add_argument(
        "--ckpt-dir", type=str, default=DEFAULT_CFG["checkpoint_dir"],
        help="Directory containing fold_N_best.pth files"
    )
    parser.add_argument(
        "--train-root", type=str, default=DEFAULT_CFG["train_root"],
        help="Training data root (used to recompute normalisation params)"
    )
    parser.add_argument(
        "--test-root", type=str, default=DEFAULT_CFG["test_root"],
        help="Test data root"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_CFG["output_path"],
        help="Path for the output submission CSV"
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(range(1, N_FOLDS + 1)),
        help="Which fold numbers to include (default: 1 2 3 4 5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_CFG["batch_size"]
    )
    args = parser.parse_args()

    cfg = dict(DEFAULT_CFG)
    cfg["checkpoint_dir"] = args.ckpt_dir
    cfg["train_root"]     = args.train_root
    cfg["test_root"]      = args.test_root
    cfg["output_path"]    = args.output
    cfg["batch_size"]     = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  HAR 1D-ResNet — 5-Fold Ensemble Inference")
    print("=" * 60)
    print(f"  Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Checkpoint dir: {os.path.abspath(cfg['checkpoint_dir'])}")
    print(f"  Folds included: {args.folds}")
    print(f"  Output path  : {os.path.abspath(cfg['output_path'])}")
    print("=" * 60)

    submission = ensemble_predict(cfg, device, folds=args.folds)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(cfg["output_path"]))
    os.makedirs(out_dir, exist_ok=True)

    submission.to_csv(cfg["output_path"], index=False)

    print(f"\n  Ensemble prediction complete.")
    print(f"  Total test samples : {len(submission)}")
    pred_dist = submission["Label"].value_counts().sort_index()
    print("  Predicted label distribution:")
    for lbl, cnt in pred_dist.items():
        print(f"    Label {lbl}: {cnt:>5d}  ({cnt / len(submission) * 100:.1f}%)")
    print(f"\n  Submission saved → {cfg['output_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
