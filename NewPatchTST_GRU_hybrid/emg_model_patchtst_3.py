# ---------------------------------------------------------
#  PatchTST → StrongGRU Hybrid Model for MFSC EMG
#  (FINAL, DEBUG-SAFE, HIGH ACCURACY VERSION)
# ---------------------------------------------------------

import torch
import torch.nn as nn
import math


# =========================================================
#  PER-CHANNEL PROJECTOR
# =========================================================
class ChannelProjector(nn.Module):
    def __init__(self, channel_dim, proj_dim):
        super().__init__()
        self.fc = nn.Linear(channel_dim, proj_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, L, channel_dim)
        return self.act(self.fc(x))


# =========================================================
# POSITIONAL ENCODING
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=600):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1,max_len,d_model)

    def forward(self, x):
        # x: (B,L,D)
        return x + self.pe[:, :x.size(1), :]


# =========================================================
#  PATCHTST ENCODER (CORRECT, CHANNEL-AWARE)
# =========================================================
class PatchTSTEncoder(nn.Module):
    """
    Input:  (B, L, 216)
    Output: (B, L, proj_dim)
    """

    def __init__(self, input_dim=216, num_channels=6,
                 proj_dim=256, layers=2, heads=4, dropout=0.1):
        super().__init__()

        assert input_dim % num_channels == 0, \
            f"Patch dim {input_dim} not divisible by {num_channels} channels"

        self.num_channels = num_channels
        self.channel_dim = input_dim // num_channels  # = 36
        self.proj_dim = proj_dim

        # 1) Per-channel projection (36 → 256)
        self.projectors = nn.ModuleList([
            ChannelProjector(self.channel_dim, proj_dim)
            for _ in range(num_channels)
        ])

        # 2) Channel fusion
        self.mixer = nn.Sequential(
            nn.Linear(num_channels * proj_dim, proj_dim),
            nn.GELU()
        )

        # 3) Positional encoding
        self.pos = PositionalEncoding(proj_dim)

        # 4) Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x):
        """
        x: (B, L, 216)
        """
        B, L, D = x.shape
        assert D == self.num_channels * self.channel_dim, \
            f"Expected {self.num_channels*self.channel_dim}, got {D}"

        # reshape into channels: (B,L,6,36)
        x = x.reshape(B, L, self.num_channels, self.channel_dim)

        # per-channel linear projection
        pcs = []
        for c in range(self.num_channels):
            pc = self.projectors[c](x[:, :, c, :])  # (B,L,256)
            pcs.append(pc)

        # concat 6 channels → 6*256 = 1536
        tokens = torch.cat(pcs, dim=2)

        # mix down to 256
        tokens = self.mixer(tokens)  # (B,L,256)

        # add positional encoding
        tokens = self.pos(tokens)

        # transformer
        encoded = self.transformer(tokens)  # (B,L,256)

        return encoded


# =========================================================
#  STRONG GRU BACKEND (IDENTICAL TO YOUR 78% MODEL)
# =========================================================
class StrongGRU(nn.Module):
    """
    BiGRU(512) × 2 layers → per-frame logits → mean over time
    """

    def __init__(self, proj_dim=256, hidden_dim=512, num_layers=2, num_classes=101):
        super().__init__()

        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)     # (B,L,1024)
        logits = self.head(out)  # (B,L,101)
        return logits


# =========================================================
#  HYBRID MODEL (PatchTST → StrongGRU)
# =========================================================
class EMG_PatchTST_GRU_Hybrid(nn.Module):
    def __init__(self, input_dim=216, num_classes=101):
        super().__init__()

        # PatchTST encoder
        self.encoder = PatchTSTEncoder(
            input_dim=input_dim,
            proj_dim=256,
            layers=2,
            heads=4,
            dropout=0.1
        )

        # GRU backend
        self.gru = StrongGRU(
            proj_dim=256,
            hidden_dim=512,
            num_layers=2,
            num_classes=num_classes
        )

    def forward(self, x):
        """
        x: (B, L, 216)
        """
        B, L, D = x.shape
        assert D == 216, f"[Hybrid] Expected patch_dim=216, got: {D}"

        encoded = self.encoder(x)     # (B,L,256)
        logits  = self.gru(encoded)   # (B,L,101)

        # mean over the time dimension (identical to your GRU baseline)
        out = logits.mean(dim=1)      # (B,101)

        return out
