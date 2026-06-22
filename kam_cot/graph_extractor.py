"""
Knowledge Graph Extractor (Module C trong spec)
================================================
Trích xuất đồ thị tri thức từ ConceptNet 5.5.

Quy trình:
  1. Trích xuất thực thể từ câu hỏi / ngữ cảnh
  2. Query ConceptNet mở rộng 1-hop và 2-hop
  3. Cắt tỉa còn tối đa p = 200 nodes
  4. Gán nhãn node bằng averaged embeddings từ Language Encoder
  5. Xây dựng edge_index và edge_type

Relation types: 17 base × 2 directions = 34 directed types
  - forward:  IsA_fwd, UsedFor_fwd, ...
  - reverse:  IsA_rev, UsedFor_rev, ...

Dùng conceptnet-lite nếu có, fallback về subgraph đơn giản.
"""

from typing import Dict, List, Optional, Callable
import torch


class ConceptNetExtractor:
    """
    Knowledge Graph extraction từ ConceptNet 5.5.
    
    Tham số:
      language   : ngôn ngữ (mặc định 'en')
      max_nodes  : tối đa số nodes (mặc định 200)
      max_hops   : số hop mở rộng (mặc định 2)
      device     : thiết bị tính toán
    """
    # 17 base relation types (theo spec — KAM-CoT paper)
    BASE_RELATIONS = [
        'IsA', 'UsedFor', 'HasA', 'CapableOf', 'AtLocation',
        'Causes', 'Desires', 'HasProperty', 'PartOf', 'CreatedBy',
        'LocatedNear', 'RelatedTo', 'SimilarTo', 'Synonym',
        'Antonym', 'DistinctFrom', 'DerivedFrom',
    ]

    def __init__(
        self,
        language: str = 'en',
        max_nodes: int = 200,
        max_hops: int = 2,
        db_path: str = "./conceptnet.db",
        device: torch.device = None,
    ):
        self.language = language
        self.max_nodes = max_nodes
        self.max_hops = max_hops
        self.db_path = db_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._connected = False

        # Map relation → index (0..33 cho 34 directed types)
        # conceptnet-lite trả về lowercase: "is_a", "at_location", ...
        # Cần map sang CamelCase: "IsA", "AtLocation", ...
        self._cn_rel_to_camel = {
            'is_a': 'IsA', 'used_for': 'UsedFor', 'has_a': 'HasA',
            'capable_of': 'CapableOf', 'at_location': 'AtLocation',
            'causes': 'Causes', 'desires': 'Desires',
            'has_property': 'HasProperty', 'part_of': 'PartOf',
            'created_by': 'CreatedBy', 'located_near': 'LocatedNear',
            'related_to': 'RelatedTo', 'similar_to': 'SimilarTo',
            'synonym': 'Synonym', 'antonym': 'Antonym',
            'distinct_from': 'DistinctFrom', 'derived_from': 'DerivedFrom',
        }

        self.rel_to_idx = {}
        for i, rel in enumerate(self.BASE_RELATIONS):
            self.rel_to_idx[f'{rel}_fwd'] = i * 2
            self.rel_to_idx[f'{rel}_rev'] = i * 2 + 1

    def extract_entities(self, text: str) -> List[str]:
        """
        Trích xuất thực thể tiềm năng từ văn bản.
        Dùng heuristics đơn giản: chọn các token dài >= 3, bỏ stopwords.
        """
        import re
        tokens = re.findall(r'[a-zA-Z]{3,}', text.lower())
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'has', 'was', 'had', 'her', 'his', 'its', 'how',
            'why', 'who', 'which', 'where', 'when', 'what', 'that',
            'this', 'with', 'from', 'they', 'been', 'have', 'will',
            'would', 'could', 'should', 'more', 'some', 'than',
            'also', 'very', 'just', 'about', 'into', 'over', 'such',
        }
        entities = [t for t in tokens if t not in stopwords]
        return list(dict.fromkeys(entities))  # unique, giữ thứ tự

    def _connect(self):
        """Kết nối conceptnet-lite database."""
        if self._connected:
            return
        import conceptnet_lite
        conceptnet_lite.connect(self.db_path)
        self._connected = True

    def _get_concept(self, name: str):
        """Lấy concept từ conceptnet-lite bằng Label lookup."""
        from conceptnet_lite import Label
        try:
            label = Label.get(text=name.lower())
            if label is None:
                return None
            # Tìm concept đúng ngôn ngữ
            for c in label.concepts:
                if c.language == self.language:
                    return c
            # Fallback: concept đầu tiên
            return label.concepts[0] if label.concepts else None
        except Exception:
            return None

    def query(self, entities: List[str]) -> Dict:
        """
        Query ConceptNet mở rộng.
        
        Returns:
            {'nodes': [node_names], 'edges': [(src, rel, tgt), ...]}
        """
        # Thử dùng conceptnet-lite
        try:
            return self._query_conceptnet_lite(entities)
        except Exception:
            return self._fallback_subgraph(entities)

    def _query_conceptnet_lite(self, entities: List[str]) -> Dict:
        """Dùng conceptnet-lite để query thật.
        
        Sử dụng API giống như trong Untitled0.ipynb:
            Label.get(text='cat').concepts[0]
            edges_for([concept], same_language=True)
            edge.relation.name, edge.start.text, edge.end.text
        """
        self._connect()
        from conceptnet_lite import edges_for

        all_nodes = set()     # node name → node object
        edges = []            # list of (start_text, rel_name, end_text)
        queue = list(entities)

        for hop in range(self.max_hops + 1):
            if not queue or len(all_nodes) >= self.max_nodes:
                break
            batch = queue
            queue = []

            for name in batch:
                norm_name = name.lower().replace(' ', '_')
                if norm_name in all_nodes or len(all_nodes) >= self.max_nodes:
                    continue

                concept = self._get_concept(norm_name)
                if concept is None:
                    continue

                all_nodes.add(norm_name)

                # Lấy edges — giống notebook: edges_for([concept], same_language=True)
                for edge in edges_for([concept], same_language=True):
                    if len(edges) >= self.max_nodes * 4:
                        break

                    # Giống notebook: edge.relation.name
                    rel_lower = edge.relation.name
                    # Giống notebook: edge.start.text, edge.end.text
                    start_text = edge.start.text.lower().replace(' ', '_')
                    end_text = edge.end.text.lower().replace(' ', '_')

                    if start_text and end_text:
                        # Map lowercase relation → CamelCase
                        rel_camel = self._cn_rel_to_camel.get(rel_lower, 'RelatedTo')
                        edges.append((start_text, rel_camel, end_text))

                        # Mở rộng queue (hop < max_hops)
                        if hop < self.max_hops:
                            if start_text not in all_nodes:
                                queue.append(start_text)
                            if end_text not in all_nodes:
                                queue.append(end_text)

        return {
            'nodes': list(all_nodes)[:self.max_nodes],
            'edges': edges,
        }

    def _fallback_subgraph(self, entities: List[str]) -> Dict:
        """
        Fallback khi không có conceptnet-lite.
        Tạo star graph từ các thực thể với quan hệ ngẫu nhiên.
        
        Note: Để chạy thật, cần cài conceptnet-lite và tải dữ liệu.
        """
        import random
        nodes = list(dict.fromkeys(entities))
        if len(nodes) < 2:
            nodes += ['ai', 'model', 'data', 'knowledge', 'reasoning']

        nodes = nodes[:self.max_nodes]
        edges = []

        # Star graph
        for i in range(1, min(len(nodes), self.max_nodes)):
            rel = random.choice(self.BASE_RELATIONS)
            edges.append((nodes[0], rel, nodes[i]))

        # Cross edges
        for i in range(1, min(len(nodes) - 1, 15)):
            rel = random.choice(self.BASE_RELATIONS)
            edges.append((nodes[i], rel, nodes[i + 1]))

        return {'nodes': nodes, 'edges': edges}

    def build_graph_data(
        self,
        text: str,
        embed_fn: Callable[[List[str]], torch.Tensor],
    ) -> Dict:
        """
        Xây dựng graph data từ text input.

        Args:
            text: Văn bản đầu vào (câu hỏi, ngữ cảnh, ...)
            embed_fn: Hàm nhúng node names → (num_nodes, d_model)
        Returns:
            dict với keys:
              - node_features: (max_nodes, d_model)
              - edge_index: (2, E)
              - edge_type: (E,)
        """
        entities = self.extract_entities(text)
        subgraph = self.query(entities)

        node_names = subgraph['nodes']
        p = len(node_names)

        # Pad/trim to max_nodes
        if p < self.max_nodes:
            node_names = node_names + ['<pad>'] * (self.max_nodes - p)
        else:
            node_names = node_names[:self.max_nodes]

        # Node embeddings
        node_features = embed_fn(node_names)  # (max_nodes, d_model)

        # Build edge_index
        node_to_idx = {n: i for i, n in enumerate(node_names)}
        src_list, tgt_list, rel_list = [], [], []

        for src_name, rel, tgt_name in subgraph['edges']:
            si = node_to_idx.get(src_name)
            ti = node_to_idx.get(tgt_name)
            if si is None or ti is None:
                continue
            if si >= self.max_nodes or ti >= self.max_nodes:
                continue
            src_list.append(si)
            tgt_list.append(ti)

            # Map relation → directed index (mặc định forward)
            rid = self.rel_to_idx.get(f'{rel}_fwd', 0)
            rel_list.append(rid)

        if not src_list:
            src_list, tgt_list, rel_list = [0], [0], [0]

        return {
            'node_features': node_features,                # (max_nodes, d_model)
            'edge_index': torch.tensor(
                [src_list, tgt_list], dtype=torch.long, device=self.device
            ),  # (2, E)
            'edge_type': torch.tensor(
                rel_list, dtype=torch.long, device=self.device
            ),  # (E,)
        }
