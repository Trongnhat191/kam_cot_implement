"""
Vision Encoder (Module B trong spec)
=====================================
Bộ mã hóa hình ảnh sử dụng DETR (mặc định) hoặc ViT.

Kiến trúc:
  1. Backbone trích xuất đặc trưng patch / feature map
  2. Linear projection chiếu về d_model (mặc định 768)

DETR path:
  - Dùng DetrModel (ResNet-101 backbone + transformer encoder)
  - Input:  (B, 3, H, W)
  - Output: (B, m, d_model) với m = số patches/queries

ViT path:
  - Dùng ViTModel
  - Input:  (B, 3, H, W)
  - Output: (B, m, d_model) với m = số patches (bỏ CLS token)

Tham khảo:
  - DETR: facebook/detr-resnet-101-dc5
  - ViT:  google/vit-base-patch16-384
"""

import math
import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """
    Vision encoder với DETR hoặc ViT backbone.
    
    Tham số (có thể thay đổi khi khởi tạo):
      d_model    : kích thước embedding đầu ra (mặc định 768)
      model_name : tên pretrained model trên HuggingFace
      use_vit    : True = dùng ViT, False = dùng DETR
    """
    def __init__(
        self,
        d_model: int = 768,
        model_name: str = "facebook/detr-resnet-101-dc5",
        use_vit: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_vit = use_vit

        if use_vit:
            from transformers import ViTModel
            self.backbone = ViTModel.from_pretrained(model_name)
            vit_dim = self.backbone.config.hidden_size
            self.projection = nn.Linear(vit_dim, d_model) if vit_dim != d_model else nn.Identity()
            self.num_patches = getattr(self.backbone.config, 'num_patches', 576)
        else:
            from transformers import DetrModel
            self.backbone = DetrModel.from_pretrained(model_name, output_hidden_states=False)
            detr_dim = self.backbone.config.d_model  # 256 for DETR
            self.projection = nn.Linear(detr_dim, d_model)
            self.num_queries = getattr(self.backbone.config, 'num_queries', 100)

        self.dropout = nn.Dropout(0.1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) — ảnh đã được normalize
        Returns:
            H_img: (B, m, d_model) — patch features
        """
        if self.use_vit:
            outputs = self.backbone(pixel_values)
            # outputs.last_hidden_state: (B, num_patches+1, vit_dim)
            patches = outputs.last_hidden_state[:, 1:, :]  # bỏ CLS token
            H_img = self.projection(patches)
        else:
            outputs = self.backbone(pixel_values)

            # DETR: ưu tiên dùng encoder output
            if outputs.encoder_last_hidden_state is not None:
                H_img_raw = outputs.encoder_last_hidden_state
            else:
                # Fallback: backbone features → positional encoding
                features = self.backbone.model.backbone(pixel_values)
                x = features[-1] if isinstance(features, (list, tuple)) else \
                    list(features.values())[0] if isinstance(features, dict) else features
                B, C, Hf, Wf = x.shape
                x = x.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)
                # Projection tạm nếu cần
                temp_proj = nn.Linear(C, self.backbone.config.d_model,
                                      bias=False).to(x.device)
                H_img_raw = temp_proj(x)
                # Thêm positional encoding
                pe = self._sinusoidal_pe(Hf * Wf, H_img_raw.size(-1), x.device)
                H_img_raw = H_img_raw + pe.unsqueeze(0)

            H_img = self.projection(H_img_raw)  # (B, m, d_model)

        H_img = self.dropout(H_img)
        return H_img

    @staticmethod
    def _sinusoidal_pe(length: int, dim: int, device: torch.device) -> torch.Tensor:
        pe = torch.zeros(length, dim, device=device)
        pos = torch.arange(length, device=device).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, device=device).float() *
                        (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe
