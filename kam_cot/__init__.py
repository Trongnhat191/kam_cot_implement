"""
KAM-CoT: Knowledge Augmented Multimodal Chain-of-Thoughts
=========================================================
Implementation for Google Colab T4 GPU.

Reference: instruction.md — Kiến trúc Mô hình KAM-CoT
"""

from .cross_attention import CrossAttention
from .gated_fusion import GatedFusion
from .vision_encoder import VisionEncoder
from .image_captioner import ImageCaptioner
from .graph_encoder import GraphEncoder
from .graph_extractor import ConceptNetExtractor
from .model import KAMCoTModel

__version__ = "1.0.0"
