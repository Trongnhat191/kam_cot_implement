"""
KAM-CoT Full Model (Module A-E + Decoder)
==========================================
Kết hợp tất cả thành phần:

  A. Language Encoder  — FLAN-T5 encoder
  B. Vision Encoder    — DETR / ViT
  C. Graph Encoder     — RGAT + GCN
  D. Cross-Attention   — Lang→Img, Lang→KG
  E. Gated Fusion      — Fusion-1
  F. Transformer Decoder — FLAN-T5 decoder

Sơ đồ luồng:
  Input Text ──► Language Encoder ──► H_lang ──┐
  Input Image ──► Vision Encoder  ──► H_img  ──┤
  Input KG   ──► Graph Encoder   ──► H_kg   ──┤
                                               │
  H_lang ── CrossAttn(Query=H_lang, KV=H_img) ──► H_img_attn
  H_lang ── CrossAttn(Query=H_lang, KV=H_kg)  ──► H_kg_attn
                                               │
  H_lang, H_img_attn, H_kg_attn ──► GatedFusion ──► H_fuse
                                                        │
  H_fuse ──► FLAN-T5 Decoder ──► Output probabilities
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from .cross_attention import CrossAttention
from .gated_fusion import GatedFusion
from .vision_encoder import VisionEncoder
from .graph_encoder import GraphEncoder


class KAMCoTModel(nn.Module):
    """
    KAM-CoT: Knowledge Augmented Multimodal Chain-of-Thoughts.
    
    Tham số (có thể thay đổi khi khởi tạo):
      model_name      : tên T5 model (mặc định google/flan-t5-base)
      vision_model    : tên vision backbone (mặc định DETR)
      use_vit         : True = ViT, False = DETR
      d_model         : kích thước embedding (auto từ T5 nếu None)
      graph_edge_dim  : kích thước edge embedding (mặc định 64)
      graph_num_rels  : số loại quan hệ (mặc định 34)
      graph_num_heads : số head RGAT (mặc định 4)
      dropout         : tỷ lệ dropout (mặc định 0.1)
    """
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        vision_model: str = "facebook/detr-resnet-101-dc5",
        use_vit: bool = False,
        d_model: int = None,
        graph_edge_dim: int = 64,
        graph_num_rels: int = 34,
        graph_num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        from transformers import T5ForConditionalGeneration

        # T5 backbone (encoder + decoder + LM head)
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.d_model = d_model or self.t5.config.d_model  # 768 cho T5-Base

        # A. Language Encoder (shared embeddings với decoder)
        self.language_encoder = self.t5.encoder

        # B. Vision Encoder
        self.vision_encoder = VisionEncoder(
            d_model=self.d_model,
            model_name=vision_model,
            use_vit=use_vit,
        )

        # C. Graph Encoder
        self.graph_encoder = GraphEncoder(
            d_model=self.d_model,
            edge_dim=graph_edge_dim,
            num_relations=graph_num_rels,
            num_heads=graph_num_heads,
            dropout=dropout,
        )

        # D. Cross-Attention Modules
        self.cross_attn_img = CrossAttention(self.d_model, dropout=dropout)
        self.cross_attn_kg = CrossAttention(self.d_model, dropout=dropout)

        # E. Gated Fusion
        self.gated_fusion = GatedFusion(self.d_model)

        # Layer norms
        self.norm_lang = nn.LayerNorm(self.d_model)
        self.norm_img = nn.LayerNorm(self.d_model)
        self.norm_kg = nn.LayerNorm(self.d_model)
        self.norm_fuse = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def encode_multimodal(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        kg_node_features: Optional[torch.Tensor] = None,
        kg_edge_index: Optional[torch.Tensor] = None,
        kg_edge_type: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Encode đa phương thức: text + image + KG → H_fuse.

        Args:
            input_ids: (B, n) token IDs
            attention_mask: (B, n)
            pixel_values: (B, 3, H, W) hoặc None
            kg_node_features: (total_nodes, d_model) hoặc None
            kg_edge_index: (2, E) hoặc None
            kg_edge_type: (E,) hoặc None

        Returns:
            dict với H_fuse và các hidden states trung gian
        """
        batch_size = input_ids.size(0)
        _debug = not hasattr(self, '_debug_done')

        # --- A. Language Encoding ---
        lang_out = self.language_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        H_lang = self.norm_lang(lang_out.last_hidden_state)  # (B, n, d)
        if _debug and H_lang.isnan().any():
            print(f"  [NaN TRACE] H_lang has NaN!")

        # --- B. Vision Encoding ---
        if pixel_values is not None:
            H_img = self.vision_encoder(pixel_values)        # (B, m, d)
            H_img = self.norm_img(H_img)
            if _debug and H_img.isnan().any():
                print(f"  [NaN TRACE] H_img has NaN!")
        else:
            H_img = torch.zeros(batch_size, 1, self.d_model,
                                device=H_lang.device, dtype=H_lang.dtype)

        # --- C. Graph Encoding ---
        if kg_node_features is not None and kg_edge_index is not None:
            total_nodes = kg_node_features.size(0)
            assert total_nodes % batch_size == 0, (
                f"kg_node_features total_nodes ({total_nodes}) not divisible by "
                f"batch_size ({batch_size}). Check collate function."
            )
            H_kg_flat = self.graph_encoder(kg_node_features, kg_edge_index, kg_edge_type)
            if _debug and H_kg_flat.isnan().any():
                print(f"  [NaN TRACE] H_kg_flat (graph encoder output) has NaN!")
            H_kg_flat = self.norm_kg(H_kg_flat)
            p = total_nodes // batch_size                       # = max_nodes
            H_kg = H_kg_flat.view(batch_size, p, self.d_model)  # (B, max_nodes, d)

        else:
            H_kg = torch.zeros(batch_size, 1, self.d_model,
                               device=H_lang.device, dtype=H_lang.dtype)

        # --- D. Cross-Attention (in float32 for stability) ---
        H_lang_f32 = H_lang.float()
        H_img_f32 = H_img.float()
        H_kg_f32 = H_kg.float()

        H_img_attn = self.cross_attn_img(query=H_lang_f32, key=H_img_f32, value=H_img_f32)
        if _debug and H_img_attn.isnan().any():
            print(f"  [NaN TRACE] H_img_attn (cross_attn_img) has NaN!")
        H_kg_attn = self.cross_attn_kg(query=H_lang_f32, key=H_kg_f32, value=H_kg_f32)
        if _debug and H_kg_attn.isnan().any():
            print(f"  [NaN TRACE] H_kg_attn (cross_attn_kg) has NaN!")

        # --- E. Gated Fusion ---
        H_fuse = self.gated_fusion(H_lang_f32, H_img_attn.float(), H_kg_attn.float())
        H_fuse = self.norm_fuse(H_fuse)

        if _debug:
            if H_fuse.isnan().any():
                print(f"  [NaN TRACE] H_fuse has NaN!")
            else:
                print(f"  [NaN TRACE] All clear — no NaN in H_fuse")
            self._debug_done = True

        return {
            'H_fuse': H_fuse,
            'H_lang': H_lang,
            'H_img': H_img,
            'H_kg': H_kg,
            'H_img_attn': H_img_attn,
            'H_kg_attn': H_kg_attn,
        }


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        kg_node_features: Optional[torch.Tensor] = None,
        kg_edge_index: Optional[torch.Tensor] = None,
        kg_edge_type: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict:
        """
        Forward pass đầy đủ.

        Args:
            input_ids: (B, n) text tokens
            attention_mask: (B, n)
            pixel_values: (B, 3, H, W) hoặc None
            kg_node_features: (B*p, d) hoặc None
            kg_edge_index: (2, E) hoặc None
            kg_edge_type: (E,) hoặc None
            labels: (B, tgt_len) — labels cho training
            decoder_input_ids: (B, tgt_len) — cho generation
        Returns:
            dict với loss, logits, hidden states
        """
        # Encode multimodal → H_fuse
        enc_out = self.encode_multimodal(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            kg_node_features=kg_node_features,
            kg_edge_index=kg_edge_index,
            kg_edge_type=kg_edge_type,
        )

        # --- F. Decoder ---
        decoder_kwargs = {}
        if labels is not None:
            decoder_kwargs['labels'] = labels
        if decoder_input_ids is not None:
            decoder_kwargs['decoder_input_ids'] = decoder_input_ids

        encoder_outputs = BaseModelOutput(
            last_hidden_state=enc_out['H_fuse'],
        )

        t5_out = self.t5(
            input_ids=None,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            return_dict=True,
            **decoder_kwargs,
        )

        return {
            'loss': t5_out.loss if labels is not None else None,
            'logits': t5_out.logits,
            **enc_out,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        kg_node_features: Optional[torch.Tensor] = None,
        kg_edge_index: Optional[torch.Tensor] = None,
        kg_edge_type: Optional[torch.Tensor] = None,
        **gen_kwargs,
    ) -> torch.Tensor:
        """
        Sinh văn bản từ multimodal input.

        Args:
            Giống encode_multimodal
            gen_kwargs: tham số cho generate() (max_length, num_beams, ...)
        Returns:
            (B, gen_len) token IDs
        """
        enc_out = self.encode_multimodal(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            kg_node_features=kg_node_features,
            kg_edge_index=kg_edge_index,
            kg_edge_type=kg_edge_type,
        )

        defaults = dict(max_length=128, num_beams=4, early_stopping=True)
        defaults.update(gen_kwargs)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=enc_out['H_fuse'],
        )

        return self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **defaults,
        )

    def get_node_embed_fn(self, tokenizer):
        """
        Tạo hàm nhúng node names bằng averaged token embeddings.
        
        Usage:
            embed_fn = model.get_node_embed_fn(tokenizer)
            embeddings = embed_fn(['dog', 'cat', ...])
        
        Returns:
            function(List[str]) → (num_nodes, d_model)
        """
        embed_tokens = self.language_encoder.get_input_embeddings()

        def _embed(node_names: List[str]) -> torch.Tensor:
            embeddings = []
            device = embed_tokens.weight.device
            for name in node_names:
                if name == '<pad>':
                    embeddings.append(torch.zeros(self.d_model, device=device))
                    continue
                tokens = tokenizer.tokenize(name.replace('_', ' '))
                if not tokens:
                    tokens = ['<unk>']
                ids = tokenizer.convert_tokens_to_ids(tokens)
                ids_t = torch.tensor(ids, device=device)
                emb = embed_tokens(ids_t).mean(dim=0)  # (d_model,)
                embeddings.append(emb)
            return torch.stack(embeddings, dim=0)

        return _embed

    def count_parameters(self) -> Dict:
        """Đếm số tham số của model."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
