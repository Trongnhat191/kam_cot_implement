"""
Gated Fusion Module — Fusion-1 (Module E trong spec)
=====================================================
Cơ chế cổng điều hướng (gated fusion) với trọng số động α, β, γ.

Công thức:
  S_α = H_lang·W1 + H_img^attn·W2 + H_kg^attn·W3
  S_β = H_lang·W4 + H_img^attn·W5 + H_kg^attn·W6
  γ'  = H_lang·W7 + H_img^attn·W8 + H_kg^attn·W9

  α_ij, β_ij, γ_ij = softmax([S_α_ij, S_β_ij, S_γ_ij])  (element-wise)

  H_fuse = α·H_lang + β·H_img^attn + γ·H_kg^attn

Kích thước:
  Tất cả đầu vào: (B, n, d)
  Đầu ra:         (B, n, d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """
    Gated fusion with 3 modality gates.
    
    Tham số:
      d_model : kích thước embedding (mặc định 768)
    """
    def __init__(self, d_model: int = 768):
        super().__init__()
        # Linear projections W1..W9 ∈ R^{d×d}
        # α gate
        self.W1 = nn.Linear(d_model, d_model, bias=False)  # lang → α
        self.W2 = nn.Linear(d_model, d_model, bias=False)  # img  → α
        self.W3 = nn.Linear(d_model, d_model, bias=False)  # kg   → α
        # β gate
        self.W4 = nn.Linear(d_model, d_model, bias=False)  # lang → β
        self.W5 = nn.Linear(d_model, d_model, bias=False)  # img  → β
        self.W6 = nn.Linear(d_model, d_model, bias=False)  # kg   → β
        # γ gate
        self.W7 = nn.Linear(d_model, d_model, bias=False)  # lang → γ
        self.W8 = nn.Linear(d_model, d_model, bias=False)  # img  → γ
        self.W9 = nn.Linear(d_model, d_model, bias=False)  # kg   → γ

    def forward(
        self,
        H_lang: torch.Tensor,      # (B, n, d)
        H_img_attn: torch.Tensor,  # (B, n, d)
        H_kg_attn: torch.Tensor,   # (B, n, d)
    ) -> torch.Tensor:
        """
        Returns:
            H_fuse: (B, n, d) fused features
        """
        # Compute gate scores
        S_alpha = self.W1(H_lang) + self.W2(H_img_attn) + self.W3(H_kg_attn)
        S_beta  = self.W4(H_lang) + self.W5(H_img_attn) + self.W6(H_kg_attn)
        S_gamma = self.W7(H_lang) + self.W8(H_img_attn) + self.W9(H_kg_attn)

        # Element-wise softmax over 3 modalities
        # Stack → (B, n, d, 3) → softmax over modality dim
        scores = torch.stack([S_alpha, S_beta, S_gamma], dim=-1)
        weights = F.softmax(scores, dim=-1)  # (B, n, d, 3)

        alpha = weights[..., 0]
        beta  = weights[..., 1]
        gamma = weights[..., 2]

        # Weighted fusion
        H_fuse = alpha * H_lang + beta * H_img_attn + gamma * H_kg_attn
        return H_fuse
