"""
model.py
--------
CRNN (CNN + Bidirectional GRU) architecture for Human Activity Recognition (HAR).

Input  : (Batch, 8, 300)  – 8 stable channels × 300 timesteps
                              [mean_x, mean_y, mean_z, std_x, std_y, std_z, mean_mag, std_mag]
Output : (Batch, num_classes)  – variable logit vector

Architecture outline (v8, base_filters=64)
──────────────────────────────────────────
  CNN Local Extractor:
    Stem:   Conv1d(8→64, k=11, s=2) → BN → ReLU → MaxPool(k=3, s=2)   [L: 300→75]
    Block1: Conv1d(64→64, k=7, s=1) → BN → ReLU → SpatialDropout(0.1)  [L: 75]
    Block2: Conv1d(64→128, k=5, s=2)→ BN → ReLU → SpatialDropout(0.2)  [L: 38]

  RNN Global Sequential Memory:
    Reshape (B, C, L) → (B, L, C)   i.e., (B, 38, 128)
    BiGRU (2 layers):
      Stage 1 Generalist: hidden_size=128   → output dim = 256 (×2 bidirectional)
      Stage 2 Specialist: hidden_size=64    → output dim = 128 (×2 bidirectional)
    Global Average Pooling across sequence length → (B, hidden_size×2)

  Classifier Head:
    Stage 1: Dropout(p=0.4) → Linear → num_classes
    Stage 2: Multi-Sample Dropout (5×, p=0.5) → 5 × Linear(shared) → num_classes
             Training loss = mean of 5 individual CE losses
             Inference logit = mean of 5 individual logits

v8 Changes
──────────
  • ResNet1D fully replaced by CRNN1D (CNN + BiGRU hybrid).
  • Stage 1 generalist: BiGRU hidden_size=128.
  • Stage 2 specialist: BiGRU hidden_size=64 + Multi-Sample Dropout.
  • Total parameters (Stage1): ~1.8 M; (Stage2): ~0.6 M  → well within 8 GB VRAM.

Regularisation (v8)
────────────────────
  • SpatialDropout1d(0.1) after CNN Block1.
  • SpatialDropout1d(0.2) after CNN Block2.
  • BiGRU internal dropout=0.3 (between layers).
  • Classifier head: Stage1 Dropout(0.4), Stage2 MultiSampleDropout(5×, p=0.5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────
# CNN Local Extractor Block
# ──────────────────────────────────────────

class ConvBlock1D(nn.Module):
    """
    Simple 1D convolution block: Conv → BN → ReLU → SpatialDropout1d.

    Parameters
    ----------
    in_channels   : int
    out_channels  : int
    kernel_size   : int
    stride        : int
    dropout_p     : float  – spatial dropout probability (channel-wise)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dropout_p: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv    = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, bias=False)
        self.bn      = nn.BatchNorm1d(out_channels)
        self.drop    = nn.Dropout1d(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(F.relu(self.bn(self.conv(x)), inplace=True))


# ──────────────────────────────────────────
# Multi-Sample Dropout Head (Stage 2)
# ──────────────────────────────────────────

class MultiSampleDropoutHead(nn.Module):
    """
    Stage 2 classifier head using Multi-Sample Dropout.

    k parallel dropout modules (each with p=dropout_p) are applied
    independently to the same feature vector, producing k separate
    logit outputs through one shared linear projection.

    Training  → returns list[Tensor] of k logit tensors (shape: each (B, num_classes))
    Inference → returns mean of k logit tensors           (shape: (B, num_classes))

    Parameters
    ----------
    in_features   : int
    num_classes   : int
    k             : int    – number of parallel dropout samples (default 5)
    dropout_p     : float  – dropout probability for each sample (default 0.5)
    """

    def __init__(self, in_features: int, num_classes: int,
                 k: int = 5, dropout_p: float = 0.5):
        super().__init__()
        self.k        = k
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout_p) for _ in range(k)])
        self.fc       = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, in_features)

        Returns
        -------
        Training  : list of k Tensors, each (B, num_classes)
        Inference : (B, num_classes)  – arithmetic mean of k logit outputs
        """
        logits_list = [self.fc(drop(x)) for drop in self.dropouts]
        if self.training:
            return logits_list                              # list of k tensors
        # Inference: stack → (k, B, C) → mean → (B, C)
        return torch.stack(logits_list, dim=0).mean(dim=0)


# ──────────────────────────────────────────
# Full CRNN (CNN + BiGRU)
# ──────────────────────────────────────────

class CRNN1D(nn.Module):
    """
    1D CRNN for HAR time-series classification.

    CNN local feature extractor feeds a 2-layer Bidirectional GRU for
    long-range temporal modelling.  Fully parametric: in_channels,
    num_classes, base_filters, gru_hidden, and use_multi_sample_dropout
    are all configurable.  Instantiate twice with independent weights to
    obtain Stage 1 (generalist) and Stage 2 (specialist) models.

    Parameters
    ----------
    in_channels             : int    – sensor channels (v8 default: 8)
    num_classes             : int    – output classes  (5 for Stage1, 1 for Stage2)
    base_filters            : int    – CNN width base  (default: 64)
    gru_hidden              : int    – BiGRU hidden size per direction
                                       Stage1 default: 128, Stage2 default: 64
    use_multi_sample_dropout: bool   – True for Stage 2 specialist head
    ms_dropout_k            : int    – number of parallel dropout samples (default 5)
    ms_dropout_p            : float  – dropout probability per sample (default 0.5)

    Intermediate shapes (base_filters=64, gru_hidden=128, input (B, 8, 300)):
      stem             → (B,  64, 75)
      cnn_block1       → (B,  64, 75)   kernel=7, stride=1
      cnn_block2       → (B, 128, 38)   kernel=5, stride=2
      transpose        → (B, 38, 128)
      bigru (2 layers) → (B, 38, 256)   hidden×2 (bidirectional)
      GAP seq          → (B, 256)
      head             → (B, num_classes)
    """

    def __init__(self,
                 in_channels:              int   = 8,
                 num_classes:              int   = 5,
                 base_filters:             int   = 64,
                 gru_hidden:               int   = 128,
                 use_multi_sample_dropout: bool  = False,
                 ms_dropout_k:             int   = 5,
                 ms_dropout_p:             float = 0.5):
        super().__init__()

        f = base_filters   # shorthand (64 in v8)

        # ── CNN Local Extractor ───────────────────────────────────────────────
        # Stem: aggressive initial downsampling
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, f,
                      kernel_size=11, stride=2, padding=5, bias=False),   # L: 300 → 150
            nn.BatchNorm1d(f),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),             # L: 150 → 75
        )

        # Block1: local low-level feature refinement (keep resolution)
        self.cnn_block1 = ConvBlock1D(f, f,         kernel_size=7, stride=1, dropout_p=0.1)  # L: 75

        # Block2: mid-level features + spatial downsampling for RNN compression
        self.cnn_block2 = ConvBlock1D(f, f * 2,     kernel_size=5, stride=2, dropout_p=0.2)  # L: 38

        # ── BiGRU Global Sequential Memory ───────────────────────────────────
        # Input to GRU: (B, L=38, C=f*2=128)
        self.bigru = nn.GRU(
            input_size   = f * 2,         # 128
            hidden_size  = gru_hidden,    # 128 (S1) or 64 (S2)
            num_layers   = 2,
            batch_first  = True,
            bidirectional= True,
            dropout      = 0.3,           # inter-layer dropout
        )

        gru_out_dim = gru_hidden * 2      # bidirectional → 256 (S1) or 128 (S2)

        # ── Classifier Head ───────────────────────────────────────────────────
        self.use_msd = use_multi_sample_dropout

        if use_multi_sample_dropout:
            # Stage 2 Multi-Sample Dropout head
            self.head = MultiSampleDropoutHead(
                in_features = gru_out_dim,
                num_classes = num_classes,
                k           = ms_dropout_k,
                dropout_p   = ms_dropout_p,
            )
        else:
            # Stage 1 standard single-dropout head
            self.head = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(gru_out_dim, num_classes),
            )

        # ── Weight Initialisation ─────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for pname, param in m.named_parameters():
                    if "weight_ih" in pname:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in pname:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in pname:
                        nn.init.zeros_(param.data)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 8, 300)

        Returns
        -------
        Standard head (Stage 1):
            logits : (B, num_classes)
        Multi-Sample Dropout head (Stage 2):
            Training  : list of 5 Tensors, each (B, num_classes)
            Inference : (B, num_classes)  – mean of 5 logits
        """
        # ── CNN feature extraction ─────────────────────────────────────────────
        x = self.stem(x)          # (B,  64, 75)
        x = self.cnn_block1(x)    # (B,  64, 75)
        x = self.cnn_block2(x)    # (B, 128, 38)

        # ── Reshape for RNN: (B, C, L) → (B, L, C) ────────────────────────────
        x = x.permute(0, 2, 1)    # (B, 38, 128)

        # ── BiGRU sequential modelling ─────────────────────────────────────────
        x, _ = self.bigru(x)      # (B, 38, gru_hidden*2)

        # ── Global average pooling across sequence dimension ───────────────────
        x = x.mean(dim=1)         # (B, gru_hidden*2)

        # ── Classifier projection ──────────────────────────────────────────────
        return self.head(x)       # see head docstring for shape details


# ──────────────────────────────────────────
# Quick sanity-check (not executed during training)
# ──────────────────────────────────────────

def _test_model():
    dummy = torch.randn(32, 8, 300)

    # Stage 1: 5-class generalist (BiGRU hidden=128, standard head)
    model_s1 = CRNN1D(in_channels=8, num_classes=5,
                      base_filters=64, gru_hidden=128,
                      use_multi_sample_dropout=False)
    model_s1.eval()
    out_s1 = model_s1(dummy)
    assert out_s1.shape == (32, 5), f"Stage1 unexpected shape: {out_s1.shape}"

    # Stage 2: 1-logit specialist (BiGRU hidden=64, Multi-Sample Dropout)
    model_s2 = CRNN1D(in_channels=8, num_classes=1,
                      base_filters=64, gru_hidden=64,
                      use_multi_sample_dropout=True,
                      ms_dropout_k=5, ms_dropout_p=0.5)

    # Inference mode
    model_s2.eval()
    out_s2_eval = model_s2(dummy)
    assert out_s2_eval.shape == (32, 1), f"Stage2 eval unexpected shape: {out_s2_eval.shape}"

    # Training mode → list of 5 logit tensors
    model_s2.train()
    out_s2_train = model_s2(dummy)
    assert isinstance(out_s2_train, list) and len(out_s2_train) == 5, \
        f"Stage2 train should return list of 5, got {type(out_s2_train)}"
    assert out_s2_train[0].shape == (32, 1), \
        f"Stage2 train list[0] unexpected shape: {out_s2_train[0].shape}"

    n1 = sum(p.numel() for p in model_s1.parameters() if p.requires_grad)
    n2 = sum(p.numel() for p in model_s2.parameters() if p.requires_grad)
    print(f"CRNN1D (v8) Stage1 — output: {out_s1.shape}  |  Params: {n1:,}")
    print(f"CRNN1D (v8) Stage2 — eval:   {out_s2_eval.shape}  |  "
          f"train: list×{len(out_s2_train)}  |  Params: {n2:,}")
    print("All shape assertions passed.")


if __name__ == "__main__":
    _test_model()
