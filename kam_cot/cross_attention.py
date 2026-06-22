"""
Cross-Attention Module (Module D trong spec)
=============================================
Single-headed cross-attention.

Công thức:
  Query = H_lang, Key = H_modal, Value = H_modal
  Output = softmax(Q @ K^T / sqrt(d)) @ V

Kích thước:
  Input:  query (B, n, d), key/value (B, m, d)
  Output: (B, n, d)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Single-headed cross-attention.
    
    Tham số có thể điều chỉnh:
      d_model : kích thước embedding (mặc định 768 cho T5-Base)
      dropout : tỷ lệ dropout (mặc định 0.1)
    """
    def __init__(self, d_model: int = 768, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,   # (B, n, d)
        key: torch.Tensor,     # (B, m, d)
        value: torch.Tensor,   # (B, m, d)
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query:   H_lang — (batch, n, d)
            key:     H_img hoặc H_kg — (batch, m, d)
            value:   giống key
            key_padding_mask: (B, m) True = pad token
        Returns:
            attended features: (batch, n, d)
        """
        Q = self.q_proj(query)                         # (B, n, d)
        K = self.k_proj(key)                           # (B, m, d)
        V = self.v_proj(value)                         # (B, m, d)

        attn = torch.matmul(Q, K.transpose(-2, -1))    # (B, n, m)
        attn = attn / self.scale

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).expand(-1, query.size(1), -1),
                float('-inf')
            )

        attn_weights = F.softmax(attn, dim=-1)          # (B, n, m)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)             # (B, n, d)
        out = self.out_proj(out)
        return out
