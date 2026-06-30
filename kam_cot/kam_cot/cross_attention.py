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
        # NaN guard: clamp inputs
        query = torch.clamp(query, min=-1e4, max=1e4)
        key = torch.clamp(key, min=-1e4, max=1e4)
        value = torch.clamp(value, min=-1e4, max=1e4)

        Q = self.q_proj(query)                         # (B, n, d)
        K = self.k_proj(key)                           # (B, m, d)
        V = self.v_proj(value)                         # (B, m, d)

        attn = torch.matmul(Q, K.transpose(-2, -1))    # (B, n, m)
        attn = attn / self.scale

        # Clamp attention scores to prevent overflow in softmax
        attn = torch.clamp(attn, min=-50, max=50)

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).expand(-1, query.size(1), -1),
                -1e9  # Use -1e9 instead of -inf for numerical stability
            )

        attn_weights = F.softmax(attn, dim=-1)          # (B, n, m)
        
        # Replace NaN in attention weights with uniform distribution
        if torch.isnan(attn_weights).any():
            m = attn_weights.size(-1)
            attn_weights = torch.where(
                torch.isnan(attn_weights),
                torch.ones_like(attn_weights) / m,
                attn_weights
            )
        
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)             # (B, n, d)
        out = self.out_proj(out)
        
        # Final clamp
        out = torch.clamp(out, min=-1e4, max=1e4)
        
        return out
