"""
Graph Encoder (Module C trong spec)
====================================
Mã hóa đồ thị tri thức gồm 2 tầng:
  1. RGAT (Relational Graph Attention) — quan hệ đa dạng
  2. GCN (Graph Convolutional Network)

Chi tiết:
  - Node features đầu vào: (batch_size, p, d_model) từ Language Encoder
  - Edge embeddings: bảng tra (34, 64) cho 34 loại quan hệ có hướng
  - RGAT layer: attention trên node có tính đến loại quan hệ
  - GCN layer: convolution đơn giản trên đồ thị

Tự động dùng PyTorch Geometric nếu có, fallback về thuần PyTorch nếu không.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGATLayer(nn.Module):
    """
    Relational Graph Attention layer.
    Dùng PyG nếu có, fallback về implementation thuần PyTorch.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_pyg = False

        try:
            from torch_geometric.nn import RGATConv
            # Try the full constructor (PyG >= 2.3 with edge_dim support).
            # Some PyG builds have a bug where unknown kwargs (e.g. 'num_heads')
            # leak into MessagePassing.__init__() and raise TypeError.
            # We catch Exception broadly here to always fall back safely.
            try:
                self.conv = RGATConv(
                    in_dim, out_dim, num_relations,
                    heads=num_heads, concat=False,
                    edge_dim=edge_dim, dropout=dropout,
                )
            except TypeError:
                # Older PyG: try without edge_dim / aggr kwargs
                try:
                    self.conv = RGATConv(
                        in_dim, out_dim, num_relations,
                        heads=num_heads, concat=False,
                        dropout=dropout,
                    )
                except TypeError:
                    raise ImportError("RGATConv incompatible with installed PyG version")
            self.use_pyg = True
        except (ImportError, Exception):
            # Fallback implementation
            head_dim = out_dim // num_heads if num_heads > 1 else out_dim
            self.W_q = nn.Linear(in_dim, out_dim, bias=False)
            self.W_k = nn.Linear(in_dim, out_dim, bias=False)
            self.W_v = nn.Linear(in_dim, out_dim, bias=False)
            self.W_o = nn.Linear(out_dim, out_dim)
            self.dropout = nn.Dropout(dropout)
            self.scale = head_dim ** 0.5

    def forward(self, x, edge_index, edge_type, edge_attr):
        if self.use_pyg:
            return self.conv(x, edge_index, edge_type, edge_attr=edge_attr)

        # Fallback: simplified attention
        N = x.size(0)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Build simple adjacency
        adj = torch.zeros(N, N, device=x.device)
        src, tgt = edge_index
        adj[src, tgt] = 1.0

        # Simplified attention (relation-agnostic)
        attn = torch.mm(Q, K.transpose(0, 1)) / self.scale
        attn = attn.masked_fill(adj == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.mm(attn, V)
        out = self.W_o(out)
        return out


class GCNLayer(nn.Module):
    """
    Graph Convolution layer.
    Dùng PyG nếu có, fallback về thuần PyTorch.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.use_pyg = False

        try:
            from torch_geometric.nn import GCNConv
            self.conv = GCNConv(in_dim, out_dim, add_self_loops=True)
            self.use_pyg = True
        except ImportError:
            self.W = nn.Linear(in_dim, out_dim, bias=False)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        if self.use_pyg:
            return self.conv(x, edge_index)

        # Fallback: simplified GCN
        N = x.size(0)
        adj = torch.zeros(N, N, device=x.device)
        src, tgt = edge_index
        adj[src, tgt] = 1.0
        adj = adj + torch.eye(N, device=x.device)
        deg = adj.sum(dim=1).pow(-0.5)
        deg[deg == float('inf')] = 0
        adj_norm = deg.unsqueeze(1) * adj * deg.unsqueeze(0)

        out = self.W(adj_norm @ x)
        out = self.dropout(out)
        return out


class GraphEncoder(nn.Module):
    """
    Graph encoder: Edge Embeddings → RGAT → GCN.
    
    Tham số (có thể thay đổi khi khởi tạo):
      d_model        : kích thước embedding (mặc định 768)
      edge_dim       : kích thước edge embedding (mặc định 64)
      num_relations  : số loại quan hệ có hướng (mặc định 34)
      num_heads      : số head trong RGAT (mặc định 4)
      dropout        : tỷ lệ dropout (mặc định 0.1)
    """
    def __init__(
        self,
        d_model: int = 768,
        edge_dim: int = 64,
        num_relations: int = 34,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_embedding = nn.Embedding(num_relations, edge_dim, padding_idx=0)

        self.rgat = RGATLayer(d_model, d_model, num_relations, edge_dim,
                              num_heads=num_heads, dropout=dropout)
        self.gcn = GCNLayer(d_model, d_model, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # (total_nodes, d_model)
        edge_index: torch.Tensor,  # (2, E) COO
        edge_type: torch.Tensor,   # (E,) loại quan hệ [0, 33]
    ) -> torch.Tensor:
        """
        Args:
            x: Node features từ Language Encoder embeddings
            edge_index: Edge indices (COO format)
            edge_type: Edge relation types
        Returns:
            (total_nodes, d_model) node features đã cập nhật
        """
        edge_attr = self.edge_embedding(edge_type)  # (E, 64)

        # Tầng 1: RGAT
        out = self.rgat(x, edge_index, edge_type, edge_attr)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        x = x + out  # skip connection

        # Tầng 2: GCN
        out = self.gcn(x, edge_index)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.dropout(out)
        x = x + out  # skip connection

        return x
