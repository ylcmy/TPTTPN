from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int = 4, mask: bool = False):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim必须能被num_heads整除"

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = self.model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, self.model_dim)
        self.k_proj = nn.Linear(model_dim, self.model_dim)
        self.v_proj = nn.Linear(model_dim, self.model_dim)
        self.out_proj = nn.Linear(self.model_dim, model_dim)

        self.scale = self.head_dim**-0.5

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        # query      (batch_size, ..., tgt_length, model_dim)
        # key, value (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # q, k, v (num_heads * batch_size, ..., length, head_dim)
        q = torch.cat(torch.split(q, self.head_dim, dim=-1), dim=0)
        k = torch.cat(torch.split(k, self.head_dim, dim=-1), dim=0)
        v = torch.cat(torch.split(v, self.head_dim, dim=-1), dim=0)

        attn = (
            torch.matmul(q, k.transpose(-2, -1)) * self.scale
        )  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn.masked_fill_(~mask, -torch.inf)

        attn = torch.softmax(attn, dim=-1)

        # 注意力加权
        out = attn @ v  # (num_heads * batch_size, ..., tgt_length, head_dim)

        # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)

        out = self.out_proj(out)
        return out
