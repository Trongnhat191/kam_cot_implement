"""
Image Captioning Module (Bổ trợ)
=================================
Sinh chú thích cho ảnh bằng ViT-GPT2.

Dùng để tạo ngữ cảnh văn bản bổ trợ từ ảnh (image captioning),
có thể dùng làm đầu vào phụ cho Language Encoder.

Model: nlpconnect/vit-gpt2-image-captioning
"""

from typing import List
import torch
import torch.nn as nn


class ImageCaptioner(nn.Module):
    """
    Image captioning với ViT-GPT2 encoder-decoder.
    
    Tham số:
      model_name : tên pretrained model trên HuggingFace
      device     : thiết bị chạy (auto-detect nếu None)
    """
    def __init__(
        self,
        model_name: str = "nlpconnect/vit-gpt2-image-captioning",
        device: torch.device = None,
    ):
        super().__init__()
        from transformers import VisionEncoderDecoderModel, AutoTokenizer

        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Cấu hình sinh
        self.model.config.max_length = 64
        self.model.config.num_beams = 4
        self.model.config.early_stopping = True

        self.model.eval()

    @torch.no_grad()
    def generate_caption(self, pixel_values: torch.Tensor) -> List[str]:
        """
        Sinh caption cho batch ảnh.

        Args:
            pixel_values: (B, 3, H, W) tensor ảnh
        Returns:
            List[str] các caption
        """
        pixel_values = pixel_values.to(self.device)
        output_ids = self.model.generate(pixel_values)
        captions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return captions
