"""
model.py
--------
InceptionTime architecture for Human Activity Recognition (HAR).

Input  : (Batch, 8, 300)  – 8 stable channels × 300 timesteps
                              [mean_x, mean_y, mean_z, std_x, std_y, std_z, mean_mag, std_mag]
Output : (Batch, num_classes)  – variable logit vector

Architecture outline (v9, nb_filters=64)
──────────────────────────────────────────
  InceptionTime Block (× N stacked):
    Input branches in parallel:
      1. Bottleneck Conv1d → Short-range Conv1d  (kernel_size=3)
      2. Bottleneck Conv1d → Mid-range  Conv1d  (kernel_size=11)
      3. Bottleneck Conv1d → Long-range Conv1d  (kernel_size=21)
      4. MaxPool1d → Bottleneck Conv1d           (passthrough branch)
    Concatenate all 4 branch outputs along channel dim → BN → ReLU

  Residual shortcut: every 3 blocks a 1×1 Conv shortcut is added
  to the skip connection (ResNet-style), helping gradient flow.

  Classifier Head:
    Global Average Pooling (B, C, L) → (B, C)
    Stage 1: Dropout(0.4) → Linear → num_classes
    Stage 2: Dropout(0.4) → Linear → 1  (BCEWithLogitsLoss)

v9 Changes
──────────
  • Completely decommissions all ResNet and CRNN/BiGRU components.
  • Introduces parallelised multi-scale InceptionTime blocks.
  • Identical structural block for Stage 1 generalist and Stage 2 specialist.
  • Residual shortcuts every 3 blocks for stable deep training.

Regularisation (v9)
────────────────────
  • BN after each inception block concatenation.
  • Dropout(0.4) in classifier head.
  • Kaiming / Xavier weight initialisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────
# Inception Block (single block)
# ──────────────────────────────────────────

class InceptionBlock1D(nn.Module):
    """
    Single multi-scale Inception block for 1-D time-series.

    Four parallel branches:
        1. bottleneck → Conv1d(k=3)
        2. bottleneck → Conv1d(k=11)
        3. bottleneck → Conv1d(k=21)
        4. MaxPool1d(k=3) → pointwise Conv1d (bottleneck passthrough)

    All branches produce `nb_filters` output channels.
    Their outputs are concatenated → (B, 4*nb_filters, L) → BN → ReLU.

    Parameters
    ----------
    in_channels : int   – input channel count
    nb_filters  : int   – output channels per branch (default 32)
    bottleneck  : int   – bottleneck reduction channels (default 32)
    """

    def __init__(self, in_channels: int,
                 nb_filters: int  = 32,
                 bottleneck: int  = 32):
        super().__init__()

        # Bottleneck that feeds the three Conv branches
        self.bottleneck_conv = nn.Conv1d(in_channels, bottleneck,
                                         kernel_size=1, bias=False)

        # Three parallel convolutional branches (applied after bottleneck)
        self.conv_short = nn.Conv1d(bottleneck, nb_filters,
                                    kernel_size=3,  padding=1,  bias=False)
        self.conv_mid   = nn.Conv1d(bottleneck, nb_filters,
                                    kernel_size=11, padding=5,  bias=False)
        self.conv_long  = nn.Conv1d(bottleneck, nb_filters,
                                    kernel_size=21, padding=10, bias=False)

        # MaxPool passthrough branch → pointwise conv
        self.maxpool    = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool  = nn.Conv1d(in_channels, nb_filters,
                                    kernel_size=1, bias=False)

        # BN applied after concatenation of all 4 branches
        self.bn         = nn.BatchNorm1d(nb_filters * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_channels, L)
        → (B, nb_filters*4, L)
        """
        bottleneck_out = self.bottleneck_conv(x)   # (B, bottleneck, L)

        b1 = self.conv_short(bottleneck_out)        # (B, nb_filters, L)
        b2 = self.conv_mid(bottleneck_out)          # (B, nb_filters, L)
        b3 = self.conv_long(bottleneck_out)         # (B, nb_filters, L)
        b4 = self.conv_pool(self.maxpool(x))       # (B, nb_filters, L)

        out = torch.cat([b1, b2, b3, b4], dim=1)   # (B, nb_filters*4, L)
        return F.relu(self.bn(out), inplace=True)


# ──────────────────────────────────────────
# InceptionTime Model
# ──────────────────────────────────────────

class InceptionTime1D(nn.Module):
    """
    InceptionTime: stacked Inception blocks with residual shortcuts.

    Shared structural backbone for both Stage 1 and Stage 2.

    Parameters
    ----------
    in_channels : int   – sensor channels input (v9 default: 8)
    num_classes : int   – output classes (5 for Stage1, 1 for Stage2)
    nb_filters  : int   – output channels per branch per block (default 32)
    bottleneck  : int   – bottleneck projection size      (default 32)
    depth       : int   – number of Inception blocks stacked (default 6)
    dropout_p   : float – classifier head dropout probability (default 0.4)

    Intermediate shapes (nb_filters=32, depth=6, input (B, 8, 300)):
      Block 0  : InceptionBlock(8  → 4×32=128)   (B, 128, 300)
      Block 1  : InceptionBlock(128→128)           (B, 128, 300)
      Block 2  : InceptionBlock(128→128) + residual shortcut from block 0 input
      Block 3  : InceptionBlock(128→128)           (B, 128, 300)
      Block 4  : InceptionBlock(128→128)           (B, 128, 300)
      Block 5  : InceptionBlock(128→128) + residual shortcut from block 3 input
      GAP (seq L=300 → 1)                         (B, 128)
      head → (B, num_classes)

    Total parameters: ~350 K (Stage1) / ~350 K (Stage2) – well within 8 GB VRAM.
    """

    def __init__(self,
                 in_channels: int   = 8,
                 num_classes: int   = 5,
                 nb_filters:  int   = 32,
                 bottleneck:  int   = 32,
                 depth:       int   = 6,
                 dropout_p:   float = 0.4):
        super().__init__()

        self.depth      = depth
        hidden_dim      = nb_filters * 4   # 128 when nb_filters=32

        # Build `depth` inception blocks
        self.inception_blocks = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            self.inception_blocks.append(
                InceptionBlock1D(ch, nb_filters=nb_filters, bottleneck=bottleneck))
            ch = hidden_dim

        # Residual shortcut projections: applied every 3 blocks
        # Shortcut maps input to the 3-block sub-sequence to hidden_dim channels
        self.shortcuts = nn.ModuleList()
        for res_start in range(0, depth, 3):
            in_ch_res = in_channels if res_start == 0 else hidden_dim
            self.shortcuts.append(nn.Sequential(
                nn.Conv1d(in_ch_res, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
            ))

        # Classifier head
        self.gap  = nn.AdaptiveAvgPool1d(1)   # (B, C, L) → (B, C, 1)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_classes),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 8, 300)

        Returns
        -------
        logits : (B, num_classes)
        """
        residual_input = x          # save for first shortcut (at block 3)
        shortcut_idx   = 0

        for block_idx, block in enumerate(self.inception_blocks):
            if block_idx % 3 == 0:
                # Save the entry point of this 3-block segment for residual
                sc_in = x

            x = block(x)

            if (block_idx + 1) % 3 == 0:
                # Apply residual shortcut at end of every 3 blocks
                sc_out = self.shortcuts[shortcut_idx](sc_in)
                x      = F.relu(x + sc_out, inplace=True)
                shortcut_idx += 1

        # Global average pooling: (B, C, L) → (B, C)
        x = self.gap(x).squeeze(-1)

        # Classifier projection
        return self.head(x)   # (B, num_classes)


# ──────────────────────────────────────────
# Quick sanity-check (not executed during training)
# ──────────────────────────────────────────

def _test_model():
    dummy = torch.randn(32, 8, 300)

    # Stage 1: 5-class generalist
    model_s1 = InceptionTime1D(in_channels=8, num_classes=5,
                                nb_filters=32, bottleneck=32, depth=6)
    model_s1.eval()
    with torch.no_grad():
        out_s1 = model_s1(dummy)
    assert out_s1.shape == (32, 5), f"Stage1 unexpected shape: {out_s1.shape}"

    # Stage 2: 1-logit specialist (BCEWithLogitsLoss)
    model_s2 = InceptionTime1D(in_channels=8, num_classes=1,
                                nb_filters=32, bottleneck=32, depth=6)
    model_s2.eval()
    with torch.no_grad():
        out_s2 = model_s2(dummy)
    assert out_s2.shape == (32, 1), f"Stage2 unexpected shape: {out_s2.shape}"

    n1 = sum(p.numel() for p in model_s1.parameters() if p.requires_grad)
    n2 = sum(p.numel() for p in model_s2.parameters() if p.requires_grad)
    print(f"InceptionTime1D (v9) Stage1 — output: {out_s1.shape}  |  Params: {n1:,}")
    print(f"InceptionTime1D (v9) Stage2 — output: {out_s2.shape}  |  Params: {n2:,}")
    print("All shape assertions passed.")


if __name__ == "__main__":
    _test_model()
