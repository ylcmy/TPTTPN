from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .MultiHeadAttention import MultiHeadAttention


class TransformerLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        mask: bool = False,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Linear(ff_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, dim: int = -2
    ) -> torch.Tensor:
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        if y is None:
            out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        else:
            y = y.transpose(dim, -2)
            out = self.attn(y, x, x)

        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out
