"""
model.py
--------
SE-InceptionTime architecture for Human Activity Recognition (HAR) – v10.

Input  : (Batch, 8, 300)  – 8 stable channels × 300 timesteps
                              [mean_x, mean_y, mean_z, std_x, std_y, std_z, mean_mag, std_mag]
Output : (Batch, 6)  – flat 6-class logit vector (L0, L1, L2, L3, L4, L5)

Architecture outline (v10, nb_filters=64)
──────────────────────────────────────────
  SE-InceptionTime Block (× N stacked):
    Input branches in parallel:
      1. Bottleneck Conv1d → Short-range Conv1d  (kernel_size=3)
      2. Bottleneck Conv1d → Mid-range  Conv1d  (kernel_size=11)
      3. Bottleneck Conv1d → Long-range Conv1d  (kernel_size=21)
      4. MaxPool1d → Bottleneck Conv1d           (passthrough branch)
    Concatenate all 4 branch outputs along channel dim → BN → ReLU
    Squeeze-and-Excitation block on the concatenated output (r=16 reduction)

  Residual shortcut: every 3 blocks a 1×1 Conv shortcut is added
  to the skip connection (ResNet-style), helping gradient flow.

  Classifier Head:
    Global Average Pooling (B, C, L) → (B, C)
    Dropout(0.4) → Linear → 6 (flat 6-class logits)

Squeeze-and-Excitation Block (1D)
───────────────────────────────────
  Squeeze: Global Average Pooling over temporal axis (B, C, L) → (B, C)
  Excitation:
    Linear(C → C//r) → ReLU
    Linear(C//r → C) → Sigmoid
  Scale: multiply channel-wise attention weights back onto feature maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────
# 1D Squeeze-and-Excitation Block
# ──────────────────────────────────────────

class SEBlock1D(nn.Module):
    """
    1D Squeeze-and-Excitation channel attention block.

    Squeeze : Global Average Pooling over the temporal axis.
    Excite  : Two-layer bottleneck MLP with ReLU + Sigmoid.
    Scale   : Re-calibrate feature maps channel-wise.

    Parameters
    ----------
    channels : int   – number of input/output feature map channels
    reduction : int  – bottleneck reduction ratio r (default 16)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        bottleneck = max(channels // reduction, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)           # (B, C, L) → (B, C, 1)
        self.fc1 = nn.Linear(channels, bottleneck, bias=True)
        self.fc2 = nn.Linear(bottleneck, channels,  bias=True)

        # Weight init
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, L)
        → (B, C, L)  – channel-wise re-calibrated feature maps
        """
        # Squeeze: (B, C, L) → (B, C, 1) → (B, C)
        s = self.gap(x).squeeze(-1)                  # (B, C)
        # Excite: two FC layers with bottleneck
        e = F.relu(self.fc1(s), inplace=True)        # (B, C//r)
        e = torch.sigmoid(self.fc2(e))               # (B, C)
        # Scale: broadcast (B, C) → (B, C, 1) and multiply
        return x * e.unsqueeze(-1)                   # (B, C, L)


# ──────────────────────────────────────────
# SE-Inception Block (single block)
# ──────────────────────────────────────────

class SEInceptionBlock1D(nn.Module):
    """
    Multi-scale Inception block with integrated 1D SE channel attention.

    Four parallel branches:
        1. bottleneck → Conv1d(k=3)
        2. bottleneck → Conv1d(k=11)
        3. bottleneck → Conv1d(k=21)
        4. MaxPool1d(k=3) → pointwise Conv1d (bottleneck passthrough)

    All branches produce `nb_filters` output channels.
    Concatenated → (B, 4*nb_filters, L) → BN → ReLU → SE block.

    Parameters
    ----------
    in_channels : int   – input channel count
    nb_filters  : int   – output channels per branch (default 64)
    bottleneck  : int   – bottleneck reduction channels (default 32)
    se_reduction : int  – SE bottleneck ratio r (default 16)
    """

    def __init__(self, in_channels: int,
                 nb_filters:   int = 64,
                 bottleneck:   int = 32,
                 se_reduction: int = 16):
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
        self.maxpool   = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, nb_filters,
                                   kernel_size=1, bias=False)

        # BN applied after concatenation of all 4 branches
        hidden_dim = nb_filters * 4
        self.bn    = nn.BatchNorm1d(hidden_dim)

        # SE block applied on the BN+ReLU output
        self.se    = SEBlock1D(hidden_dim, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_channels, L)
        → (B, nb_filters*4, L)  – after BN + ReLU + SE
        """
        bottleneck_out = self.bottleneck_conv(x)    # (B, bottleneck, L)

        b1 = self.conv_short(bottleneck_out)         # (B, nb_filters, L)
        b2 = self.conv_mid(bottleneck_out)           # (B, nb_filters, L)
        b3 = self.conv_long(bottleneck_out)          # (B, nb_filters, L)
        b4 = self.conv_pool(self.maxpool(x))        # (B, nb_filters, L)

        out = torch.cat([b1, b2, b3, b4], dim=1)    # (B, nb_filters*4, L)
        out = F.relu(self.bn(out), inplace=True)     # (B, nb_filters*4, L)
        out = self.se(out)                           # (B, nb_filters*4, L)  SE attention
        return out


# ──────────────────────────────────────────
# SE-InceptionTime Model (v10, flat 6-class)
# ──────────────────────────────────────────

class InceptionTime1D(nn.Module):
    """
    SE-InceptionTime: stacked SE-Inception blocks with residual shortcuts.

    v10 flat 6-class global classifier replacing the v9 hierarchical two-stage design.
    Each Inception block is followed by a 1D Squeeze-and-Excitation channel
    attention block for adaptive feature recalibration.

    Parameters
    ----------
    in_channels  : int   – sensor channels input (v10 default: 8)
    num_classes  : int   – output classes (v10 flat: 6)
    nb_filters   : int   – output channels per branch per block (default 64)
    bottleneck   : int   – bottleneck projection size           (default 32)
    depth        : int   – number of SE-Inception blocks stacked (default 6)
    dropout_p    : float – classifier head dropout probability   (default 0.4)
    se_reduction : int   – SE channel reduction ratio r          (default 16)

    Intermediate shapes (nb_filters=64, depth=6, input (B, 8, 300)):
      Block 0  : SEInceptionBlock(8   → 4×64=256) + SE   (B, 256, 300)
      Block 1  : SEInceptionBlock(256 → 256) + SE          (B, 256, 300)
      Block 2  : SEInceptionBlock(256 → 256) + SE  + residual shortcut
      Block 3  : SEInceptionBlock(256 → 256) + SE          (B, 256, 300)
      Block 4  : SEInceptionBlock(256 → 256) + SE          (B, 256, 300)
      Block 5  : SEInceptionBlock(256 → 256) + SE  + residual shortcut
      GAP (seq L=300 → 1)                                  (B, 256)
      Dropout(0.4) → Linear(256 → 6)                       (B, 6)
    """

    def __init__(self,
                 in_channels:  int   = 8,
                 num_classes:  int   = 6,
                 nb_filters:   int   = 64,
                 bottleneck:   int   = 32,
                 depth:        int   = 6,
                 dropout_p:    float = 0.4,
                 se_reduction: int   = 16):
        super().__init__()

        self.depth      = depth
        hidden_dim      = nb_filters * 4   # 256 when nb_filters=64

        # Build `depth` SE-Inception blocks
        self.inception_blocks = nn.ModuleList()
        ch = in_channels
        for _ in range(depth):
            self.inception_blocks.append(
                SEInceptionBlock1D(ch,
                                   nb_filters=nb_filters,
                                   bottleneck=bottleneck,
                                   se_reduction=se_reduction))
            ch = hidden_dim

        # Residual shortcut projections: applied every 3 blocks
        # Shortcut maps the 3-block segment input to hidden_dim channels
        self.shortcuts = nn.ModuleList()
        for res_start in range(0, depth, 3):
            in_ch_res = in_channels if res_start == 0 else hidden_dim
            self.shortcuts.append(nn.Sequential(
                nn.Conv1d(in_ch_res, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
            ))

        # Classifier head – flat 6-class
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
        logits : (B, num_classes)  – raw 6-class logits
        """
        shortcut_idx = 0

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

        # Flat 6-class projection
        return self.head(x)   # (B, num_classes)


# ──────────────────────────────────────────
# Quick sanity-check (not executed during training)
# ──────────────────────────────────────────

def _test_model():
    dummy = torch.randn(32, 8, 300)

    # v10: flat 6-class SE-InceptionTime
    model_v10 = InceptionTime1D(in_channels=8, num_classes=6,
                                 nb_filters=64, bottleneck=32, depth=6,
                                 se_reduction=16)
    model_v10.eval()
    with torch.no_grad():
        out = model_v10(dummy)
    assert out.shape == (32, 6), f"v10 unexpected output shape: {out.shape}"

    n_params = sum(p.numel() for p in model_v10.parameters() if p.requires_grad)
    print(f"SE-InceptionTime1D (v10) — output: {out.shape}  |  Params: {n_params:,}")
    print("All shape assertions passed.")


if __name__ == "__main__":
    _test_model()
