"""
model.py
--------
1D-ResNet architecture for Human Activity Recognition (HAR).

Input  : (Batch, 4, 300)  – 4 pruned channels × 300 timesteps
                              [mean_x, mean_y, mean_z, std_mag]
Output : (Batch, 6)        – 6-class logit vector

Architecture outline
────────────────────
  Stem:  Conv1d(4→32, k=11, s=2) → BN → ReLU → MaxPool(k=3, s=2)
  Stage 1: ResBlock(32→32)  × 2
  Stage 2: ResBlock(32→64)  × 2  [stride-2 downsample]
  Spatial Dropout1d(p=0.2)        ← inserted between Stage 2 and Stage 3
  Stage 3: ResBlock(64→128) × 2  [stride-2 downsample]
  Spatial Dropout1d(p=0.3)        ← inserted between Stage 3 and Stage 4
  Stage 4: ResBlock(128→256)× 2  [stride-2 downsample]
  Head:  Global Average Pooling → Dropout(p=0.4) → FC(256→6)

Architectural changes (v6)
──────────────────────────
  • in_channels: 8 → 4  (radical feature pruning: only mean_x/y/z + std_mag retained)
  • base_filters: 64 → 32  (halved network width to match compressed feature space)
    Filter cadence: Stem=32, Stage1=32, Stage2=64, Stage3=128, Stage4=256
  • Total parameters reduced from ~3.4 M → ~0.85 M  (prevents overfitting on
    the pruned 4-channel input space)
  • Added second spatial dropout (Dropout1d, p=0.3) between Stage 3 and Stage 4
    to compensate for the reduced network depth with stronger regularisation.

Regularisation (v6)
────────────────────
  • Classifier head dropout at p=0.4 (unchanged from v3/v5).
  • Spatial Dropout1d(p=0.2) between Stage 2 and Stage 3 (unchanged).
  • Spatial Dropout1d(p=0.3) between Stage 3 and Stage 4  (new, v6).

Total parameters: ~0.85 M  (fits comfortably in 8 GB VRAM even with large batches)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────
# Building block
# ──────────────────────────────────────────

class ResBlock1D(nn.Module):
    """
    Basic 1-D Residual Block.

        x ──► Conv(k=3) ─► BN ─► ReLU ─► Conv(k=3) ─► BN ──► (+) ─► ReLU
              └─────────────────── shortcut (1×1 Conv + BN if needed) ──────┘

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    stride       : int  – applied to the first conv; shortcut uses matching stride
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        # Identity shortcut or projection shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


# ──────────────────────────────────────────
# Full 1D-ResNet
# ──────────────────────────────────────────

class ResNet1D(nn.Module):
    """
    1D ResNet for HAR time-series classification.

    Parameters
    ----------
    in_channels  : int  – number of sensor channels (default 4: mean_x/y/z + std_mag)
    num_classes  : int  – number of output classes  (default 6)
    base_filters : int  – width multiplier for the residual stages (default 32)

    Filter cadence with base_filters=32:
      Stem=32 → Stage1=32 → Stage2=64 → Stage3=128 → Stage4=256
    """

    def __init__(self,
                 in_channels: int  = 4,
                 num_classes: int  = 6,
                 base_filters: int = 32):
        super().__init__()

        f = base_filters  # shorthand  (32 in v6)

        # ── Stem ──────────────────────────────────────────────────────────────
        # Large receptive field k=11 to capture broad temporal context at the
        # very first layer (covers ~11 consecutive time-steps).
        # in_channels=4 (v6): mean_x, mean_y, mean_z, std_mag.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, f,
                      kernel_size=11, stride=2, padding=5, bias=False),   # L: 300 → 150
            nn.BatchNorm1d(f),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),             # L: 150 → 75
        )

        # ── Residual Stages ───────────────────────────────────────────────────
        # Stage 1 – keep spatial resolution, 2 blocks  [32 → 32]
        self.stage1 = nn.Sequential(
            ResBlock1D(f,     f,     stride=1),
            ResBlock1D(f,     f,     stride=1),
        )                                                                  # L: 75

        # Stage 2 – double channels, halve length, 2 blocks  [32 → 64]
        self.stage2 = nn.Sequential(
            ResBlock1D(f,     f * 2, stride=2),
            ResBlock1D(f * 2, f * 2, stride=1),
        )                                                                  # L: 38

        # Spatial Dropout between Stage 2 and Stage 3.
        # nn.Dropout1d zeros entire channels (feature maps) along the temporal
        # dimension, providing stronger regularisation than element-wise dropout
        # and preventing the model from over-relying on any single feature map.
        # Retained at p=0.2 (unchanged from v3/v5).
        self.spatial_dropout_2 = nn.Dropout1d(p=0.2)

        # Stage 3 – double channels, halve length, 2 blocks  [64 → 128]
        self.stage3 = nn.Sequential(
            ResBlock1D(f * 2, f * 4, stride=2),
            ResBlock1D(f * 4, f * 4, stride=1),
        )                                                                  # L: 19

        # Spatial Dropout between Stage 3 and Stage 4 (new in v6).
        # p=0.3 — slightly elevated to compensate for the lighter overall
        # network capacity and encourage feature diversity in the final stage.
        self.spatial_dropout_3 = nn.Dropout1d(p=0.3)

        # Stage 4 – double channels, halve length, 2 blocks  [128 → 256]
        self.stage4 = nn.Sequential(
            ResBlock1D(f * 4, f * 8, stride=2),
            ResBlock1D(f * 8, f * 8, stride=1),
        )                                                                  # L: 10

        # ── Head ──────────────────────────────────────────────────────────────
        # Global Average Pooling: collapses temporal dimension → (B, f*8)
        # With base_filters=32: f*8 = 256
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Dropout at p=0.4 (retained from v3/v5) before the classifier to
        # prevent over-suppression of highly abstracted features in the
        # final projection.
        self.dropout = nn.Dropout(p=0.4)

        # Final fully-connected classifier: 256 → 6
        self.fc = nn.Linear(f * 8, num_classes)

        # ── Weight Initialisation ─────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 4, 300)
              4 pruned channels × 300 timesteps

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)

        Intermediate shapes (base_filters=32):
          stem          → (B,  32, 75)
          stage1        → (B,  32, 75)
          stage2        → (B,  64, 38)
          dropout_2(p=0.2) applied
          stage3        → (B, 128, 19)
          dropout_3(p=0.3) applied
          stage4        → (B, 256, 10)
          GAP           → (B, 256,  1)
          squeeze       → (B, 256)
          dropout(p=0.4)
          fc            → (B,   6)
        """
        x = self.stem(x)                   # (B,  32, 75)
        x = self.stage1(x)                 # (B,  32, 75)
        x = self.stage2(x)                 # (B,  64, 38)
        x = self.spatial_dropout_2(x)      # (B,  64, 38) – drop whole channels (p=0.2)
        x = self.stage3(x)                 # (B, 128, 19)
        x = self.spatial_dropout_3(x)      # (B, 128, 19) – drop whole channels (p=0.3)
        x = self.stage4(x)                 # (B, 256, 10)
        x = self.gap(x)                    # (B, 256,  1)
        x = x.squeeze(-1)                  # (B, 256)
        x = self.dropout(x)                # (B, 256) – p=0.4
        x = self.fc(x)                     # (B,   6)
        return x


# ──────────────────────────────────────────
# Quick sanity-check (not executed during training)
# ──────────────────────────────────────────

def _test_model():
    model = ResNet1D(in_channels=4, num_classes=6, base_filters=32)
    dummy = torch.randn(32, 4, 300)
    out   = model(dummy)
    assert out.shape == (32, 6), f"Unexpected output shape: {out.shape}"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet1D (v6) — output shape: {out.shape}  |  "
          f"Trainable params: {n_params:,}")
    return model


if __name__ == "__main__":
    _test_model()
