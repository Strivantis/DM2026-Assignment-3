"""
model.py
--------
1D-ResNet architecture for Human Activity Recognition (HAR).

Input  : (Batch, 6, 300)   – 6 channels × 300 timesteps
Output : (Batch, 6)        – 6-class logit vector

Architecture outline
────────────────────
  Stem:  Conv1d(6→64, k=11, s=2) → BN → ReLU → MaxPool(k=3, s=2)
  Stage 1: ResBlock(64→64)  × 2
  Stage 2: ResBlock(64→128) × 2  [stride-2 downsample]
  Stage 3: ResBlock(128→256)× 2  [stride-2 downsample]
  Stage 4: ResBlock(256→512)× 2  [stride-2 downsample]
  Head:  Global Average Pooling → FC(512→6)

Total parameters: ~3.5 M  (fits comfortably in 8 GB VRAM even with large batches)
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
    in_channels  : int  – number of sensor channels (default 6)
    num_classes  : int  – number of output classes  (default 6)
    base_filters : int  – width multiplier for the residual stages (default 64)
    """

    def __init__(self,
                 in_channels: int  = 6,
                 num_classes: int  = 6,
                 base_filters: int = 64):
        super().__init__()

        f = base_filters  # shorthand

        # ── Stem ──────────────────────────────────────────────────────────────
        # Large receptive field k=11 to capture broad temporal context at the
        # very first layer (covers ~11 consecutive time-steps).
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, f,
                      kernel_size=11, stride=2, padding=5, bias=False),   # L: 300 → 150
            nn.BatchNorm1d(f),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),             # L: 150 → 75
        )

        # ── Residual Stages ───────────────────────────────────────────────────
        # Stage 1 – keep spatial resolution, 2 blocks
        self.stage1 = nn.Sequential(
            ResBlock1D(f,     f,     stride=1),
            ResBlock1D(f,     f,     stride=1),
        )                                                                  # L: 75

        # Stage 2 – double channels, halve length, 2 blocks
        self.stage2 = nn.Sequential(
            ResBlock1D(f,     f * 2, stride=2),
            ResBlock1D(f * 2, f * 2, stride=1),
        )                                                                  # L: 38

        # Stage 3 – double channels, halve length, 2 blocks
        self.stage3 = nn.Sequential(
            ResBlock1D(f * 2, f * 4, stride=2),
            ResBlock1D(f * 4, f * 4, stride=1),
        )                                                                  # L: 19

        # Stage 4 – double channels, halve length, 2 blocks
        self.stage4 = nn.Sequential(
            ResBlock1D(f * 4, f * 8, stride=2),
            ResBlock1D(f * 8, f * 8, stride=1),
        )                                                                  # L: 10

        # ── Head ──────────────────────────────────────────────────────────────
        # Global Average Pooling: collapses temporal dimension → (B, f*8)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Optional lightweight dropout before the classifier
        self.dropout = nn.Dropout(p=0.3)

        # Final fully-connected classifier
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
        x : torch.Tensor, shape (B, 6, 300)

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)
        """
        x = self.stem(x)     # (B, 64, 75)
        x = self.stage1(x)   # (B, 64, 75)
        x = self.stage2(x)   # (B, 128, 38)
        x = self.stage3(x)   # (B, 256, 19)
        x = self.stage4(x)   # (B, 512, 10)
        x = self.gap(x)      # (B, 512, 1)
        x = x.squeeze(-1)    # (B, 512)
        x = self.dropout(x)
        x = self.fc(x)       # (B, 6)
        return x


# ──────────────────────────────────────────
# Quick sanity-check (not executed during training)
# ──────────────────────────────────────────

def _test_model():
    model = ResNet1D(in_channels=6, num_classes=6)
    dummy = torch.randn(32, 6, 300)
    out   = model(dummy)
    assert out.shape == (32, 6), f"Unexpected output shape: {out.shape}"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet1D — output shape: {out.shape}  |  "
          f"Trainable params: {n_params:,}")
    return model


if __name__ == "__main__":
    _test_model()
