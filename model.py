"""
model.py
--------
1D-ResNet architecture for Human Activity Recognition (HAR).

Input  : (Batch, 8, 300)  – 8 stable channels × 300 timesteps
                              [mean_x, mean_y, mean_z, std_x, std_y, std_z, mean_mag, std_mag]
Output : (Batch, num_classes)  – variable logit vector

Architecture outline (v7, base_filters=64)
──────────────────────────────────────────
  Stem:  Conv1d(8→64, k=11, s=2) → BN → ReLU → MaxPool(k=3, s=2)
  Stage 1: ResBlock(64→64)   × 2
  Stage 2: ResBlock(64→128)  × 2  [stride-2 downsample]
  Spatial Dropout1d(p=0.2)         ← between Stage 2 and Stage 3
  Stage 3: ResBlock(128→256) × 2  [stride-2 downsample]
  Spatial Dropout1d(p=0.3)         ← between Stage 3 and Stage 4
  Stage 4: ResBlock(256→512) × 2  [stride-2 downsample]
  Head:  Global Average Pooling → Dropout(p=0.4) → FC(512→num_classes)

v7 Changes
──────────
  • in_channels: 4 → 8  (restored stable 8-channel baseline; v6 pruning reverted)
  • base_filters: 32 → 64  (restored to match the wider 8-channel feature space)
    Filter cadence: Stem=64, Stage1=64, Stage2=128, Stage3=256, Stage4=512
  • num_classes is fully parametric (5 for Stage 1 generalist, 2 for Stage 2 specialist)
  • ResNet1D is instantiated twice independently (Model_Stage1 / Model_Stage2)
    with non-shared weights; each instance is a complete, standalone network.
  • Total parameters: ~3.4 M  (comfortably within 8 GB VRAM at batch_size = 128)

Regularisation (v7)
────────────────────
  • Classifier head dropout at p=0.4.
  • Spatial Dropout1d(p=0.2) between Stage 2 and Stage 3.
  • Spatial Dropout1d(p=0.3) between Stage 3 and Stage 4.
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

    Fully parametric: in_channels, num_classes, and base_filters are all
    configurable.  Instantiate twice with identical hyperparameters but
    independent weight tensors to obtain the Stage 1 (generalist) and
    Stage 2 (specialist) models.

    Parameters
    ----------
    in_channels  : int  – number of sensor channels (v7 default: 8)
    num_classes  : int  – number of output classes  (5 for Stage1, 2 for Stage2)
    base_filters : int  – width multiplier (v7 default: 64)

    Filter cadence with base_filters=64:
      Stem=64 → Stage1=64 → Stage2=128 → Stage3=256 → Stage4=512

    Intermediate shapes (base_filters=64, input (B, 8, 300)):
      stem          → (B,  64, 75)
      stage1        → (B,  64, 75)
      stage2        → (B, 128, 38)
      dropout_2(p=0.2) applied
      stage3        → (B, 256, 19)
      dropout_3(p=0.3) applied
      stage4        → (B, 512, 10)
      GAP           → (B, 512,  1)
      squeeze       → (B, 512)
      dropout(p=0.4)
      fc            → (B, num_classes)
    """

    def __init__(self,
                 in_channels:  int = 8,
                 num_classes:  int = 5,
                 base_filters: int = 64):
        super().__init__()

        f = base_filters  # shorthand  (64 in v7)

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, f,
                      kernel_size=11, stride=2, padding=5, bias=False),   # L: 300 → 150
            nn.BatchNorm1d(f),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),             # L: 150 → 75
        )

        # ── Residual Stages ───────────────────────────────────────────────────
        # Stage 1 – keep spatial resolution, 2 blocks  [64 → 64]
        self.stage1 = nn.Sequential(
            ResBlock1D(f,     f,     stride=1),
            ResBlock1D(f,     f,     stride=1),
        )                                                                  # L: 75

        # Stage 2 – double channels, halve length, 2 blocks  [64 → 128]
        self.stage2 = nn.Sequential(
            ResBlock1D(f,     f * 2, stride=2),
            ResBlock1D(f * 2, f * 2, stride=1),
        )                                                                  # L: 38

        # Spatial Dropout between Stage 2 and Stage 3 (p=0.2)
        self.spatial_dropout_2 = nn.Dropout1d(p=0.2)

        # Stage 3 – double channels, halve length, 2 blocks  [128 → 256]
        self.stage3 = nn.Sequential(
            ResBlock1D(f * 2, f * 4, stride=2),
            ResBlock1D(f * 4, f * 4, stride=1),
        )                                                                  # L: 19

        # Spatial Dropout between Stage 3 and Stage 4 (p=0.3)
        self.spatial_dropout_3 = nn.Dropout1d(p=0.3)

        # Stage 4 – double channels, halve length, 2 blocks  [256 → 512]
        self.stage4 = nn.Sequential(
            ResBlock1D(f * 4, f * 8, stride=2),
            ResBlock1D(f * 8, f * 8, stride=1),
        )                                                                  # L: 10

        # ── Head ──────────────────────────────────────────────────────────────
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc      = nn.Linear(f * 8, num_classes)

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
        x : torch.Tensor, shape (B, 8, 300)

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)
        """
        x = self.stem(x)                   # (B,  64, 75)
        x = self.stage1(x)                 # (B,  64, 75)
        x = self.stage2(x)                 # (B, 128, 38)
        x = self.spatial_dropout_2(x)      # (B, 128, 38)
        x = self.stage3(x)                 # (B, 256, 19)
        x = self.spatial_dropout_3(x)      # (B, 256, 19)
        x = self.stage4(x)                 # (B, 512, 10)
        x = self.gap(x)                    # (B, 512,  1)
        x = x.squeeze(-1)                  # (B, 512)
        x = self.dropout(x)                # (B, 512)
        x = self.fc(x)                     # (B, num_classes)
        return x


# ──────────────────────────────────────────
# Quick sanity-check (not executed during training)
# ──────────────────────────────────────────

def _test_model():
    # Stage 1: 5-class generalist
    model_s1 = ResNet1D(in_channels=8, num_classes=5, base_filters=64)
    dummy    = torch.randn(32, 8, 300)
    out_s1   = model_s1(dummy)
    assert out_s1.shape == (32, 5), f"Stage1 unexpected shape: {out_s1.shape}"

    # Stage 2: 2-class specialist
    model_s2 = ResNet1D(in_channels=8, num_classes=2, base_filters=64)
    out_s2   = model_s2(dummy)
    assert out_s2.shape == (32, 2), f"Stage2 unexpected shape: {out_s2.shape}"

    n1 = sum(p.numel() for p in model_s1.parameters() if p.requires_grad)
    n2 = sum(p.numel() for p in model_s2.parameters() if p.requires_grad)
    print(f"ResNet1D (v7) Stage1 — output: {out_s1.shape}  |  Params: {n1:,}")
    print(f"ResNet1D (v7) Stage2 — output: {out_s2.shape}  |  Params: {n2:,}")


if __name__ == "__main__":
    _test_model()
