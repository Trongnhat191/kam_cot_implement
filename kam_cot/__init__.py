"""
KAM-CoT: Knowledge Augmented Multimodal Chain-of-Thoughts
=========================================================
Implementation for Google Colab T4 GPU.

Reference: instruction.md — Kiến trúc Mô hình KAM-CoT
"""

from .kam_cot.cross_attention import CrossAttention
from .kam_cot.gated_fusion import GatedFusion
from .kam_cot.vision_encoder import VisionEncoder
from .kam_cot.image_captioner import ImageCaptioner
from .kam_cot.graph_encoder import GraphEncoder
from .kam_cot.graph_extractor import ConceptNetExtractor
from .kam_cot.model import KAMCoTModel

__version__ = "1.0.0"
