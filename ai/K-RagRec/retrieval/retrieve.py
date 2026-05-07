"""
知识子图检索与重排序模块
实现从知识图谱中检索相关知识子图，并对结果进行重排序
"""

import os
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

try:
    from ..indexing.index_KG import KnowledgeGraph
except ImportError:
    from indexing.index_KG import KnowledgeGraph


@dataclass
class Subgraph:
    """知识子图数据结构"""
    center_node: int
    nodes: List[int]
    edges: List[Tuple[int, int, int]]
    node_features: Optional[torch.Tensor] = None
    
    def to_data(self) -> Data:
        """转换为 PyG Data 对象"""
        if len(self.edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
        else:
            src = [e[0] for e in self.edges]
            dst = [e[1] for e in self.edges]
            types = [e[2] for e in self.edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_type = torch.tensor(types, dtype=torch.long)
        
        return Data(
            x=self.node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            center_node=self.center_node,
            nodes=self.nodes
        )


class GraphRetrieval:
    """
    知识子图检索器
    
    对应论文模块3和4：
    - 知识子图检索：从知识图谱中检索与查询相关的子图
    - 知识子图重排序：对检索结果进行精细排序
    
    支持两种粒度的子图：
    - 细粒度：一阶邻居子图
    - 粗粒度：二阶邻居子图
    """
    
    def __init__(
        self,
        kg: KnowledgeGraph,
        node_embeddings: torch.Tensor,
        gnn_embeddings: Optional[Dict[int, torch.Tensor]] = None,
        device: str = "cuda"
    ):
        self.kg = kg
        self.node_embeddings = node_embeddings.to(device)
        self.gnn_embeddings = gnn_embeddings
        if gnn_embeddings:
            self.gnn_embeddings = {k: v.to(device) for k, v in gnn_embeddings.items()}
        self.device = device
        
        self._build_adjacency()
    
    def _build_adjacency(self):
        """构建邻接表，加速子图提取"""
        self.adjacency: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(self.kg.num_nodes)}
        
        edge_index = self.kg.edge_index.cpu().numpy()
        edge_type = self.kg.edge_type.cpu().numpy()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            rel = edge_type[i]
            self.adjacency[src].append((dst, rel))
    
    def retrieve_topk_nodes_from_graph(
        self,
        query_embedding: torch.Tensor,
        topk_nodes: int = 10,
        use_gnn: bool = False,
        layer: int = 2
    ) -> List[int]:
        """
        从知识图谱中检索 top-k 个最相似的节点
        
        Args:
            query_embedding: 查询嵌入 [embedding_dim]
            topk_nodes: 返回的节点数量
            use_gnn: 是否使用 GNN 增强的嵌入
            layer: GNN 层索引
        
        Returns:
            节点索引列表
        """
        query_embedding = query_embedding.to(self.device)
        
        if use_gnn and self.gnn_embeddings and layer in self.gnn_embeddings:
            node_embs = self.gnn_embeddings[layer]
        else:
            node_embs = self.node_embeddings
        
        scores = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            node_embs,
            dim=-1
        )
        
        _, indices = torch.topk(scores, min(topk_nodes, scores.size(0)))
        
        return indices.cpu().tolist()
    
    def get_first_order_subgraph(self, node_idx: int) -> Subgraph:
        """
        获取一阶邻居子图（细粒度）
        
        Args:
            node_idx: 中心节点索引
        
        Returns:
            子图对象
        """
        nodes = {node_idx}
        edges = []
        
        for neighbor, rel in self.adjacency.get(node_idx, []):
            nodes.add(neighbor)
            edges.append((node_idx, neighbor, rel))
        
        node_list = list(nodes)
        node_features = self.node_embeddings[node_list] if self.node_embeddings is not None else None
        
        return Subgraph(
            center_node=node_idx,
            nodes=node_list,
            edges=edges,
            node_features=node_features
        )
    
    def get_second_order_subgraph(self, node_idx: int) -> Subgraph:
        """
        获取二阶邻居子图（粗粒度）
        
        Args:
            node_idx: 中心节点索引
        
        Returns:
            子图对象
        """
        nodes = {node_idx}
        edges = []
        
        for neighbor1, rel1 in self.adjacency.get(node_idx, []):
            nodes.add(neighbor1)
            edges.append((node_idx, neighbor1, rel1))
            
            for neighbor2, rel2 in self.adjacency.get(neighbor1, []):
                if neighbor2 not in nodes:
                    nodes.add(neighbor2)
                edges.append((neighbor1, neighbor2, rel2))
        
        node_list = list(nodes)
        node_features = self.node_embeddings[node_list] if self.node_embeddings is not None else None
        
        return Subgraph(
            center_node=node_idx,
            nodes=node_list,
            edges=edges,
            node_features=node_features
        )
    
    def retrieve_subgraphs(
        self,
        query_embedding: torch.Tensor,
        topk_nodes: int = 10,
        subgraph_order: int = 1,
        use_gnn: bool = False,
        layer: int = 2
    ) -> List[Subgraph]:
        """
        检索相关知识子图
        
        Args:
            query_embedding: 查询嵌入
            topk_nodes: 检索的节点数量
            subgraph_order: 子图阶数 (1 或 2)
            use_gnn: 是否使用 GNN 嵌入
            layer: GNN 层索引
        
        Returns:
            子图列表
        """
        node_indices = self.retrieve_topk_nodes_from_graph(
            query_embedding, topk_nodes, use_gnn, layer
        )
        
        subgraphs = []
        for node_idx in node_indices:
            if subgraph_order == 1:
                subgraph = self.get_first_order_subgraph(node_idx)
            else:
                subgraph = self.get_second_order_subgraph(node_idx)
            subgraphs.append(subgraph)
        
        return subgraphs
    
    def re_ranking(
        self,
        query_embedding: torch.Tensor,
        retrieved_nodes: List[int],
        topk_nodes: int = 5,
        use_gnn: bool = True,
        layer: int = 2
    ) -> List[int]:
        """
        对检索到的节点进行重排序
        
        使用 GNN 融合后的嵌入重新计算相似度，
        确保最相关的结果排在前面
        
        Args:
            query_embedding: 查询嵌入
            retrieved_nodes: 初步检索到的节点列表
            topk_nodes: 重排序后返回的节点数量
            use_gnn: 是否使用 GNN 嵌入
            layer: GNN 层索引
        
        Returns:
            重排序后的节点索引列表
        """
        if not retrieved_nodes:
            return []
        
        query_embedding = query_embedding.to(self.device)
        
        if use_gnn and self.gnn_embeddings and layer in self.gnn_embeddings:
            node_embs = self.gnn_embeddings[layer]
        else:
            node_embs = self.node_embeddings
        
        node_indices_tensor = torch.tensor(retrieved_nodes, device=self.device)
        candidate_embs = node_embs[node_indices_tensor]
        
        scores = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            candidate_embs,
            dim=-1
        )
        
        _, sorted_indices = torch.sort(scores, descending=True)
        
        topk = min(topk_nodes, len(retrieved_nodes))
        reranked_nodes = [retrieved_nodes[i] for i in sorted_indices[:topk].tolist()]
        
        return reranked_nodes
    
    def retrieve_and_rerank(
        self,
        query_embedding: torch.Tensor,
        topk_retrieve: int = 10,
        topk_rerank: int = 5,
        subgraph_order: int = 1,
        use_gnn_for_retrieve: bool = False,
        use_gnn_for_rerank: bool = True,
        layer: int = 2
    ) -> Tuple[List[int], List[Subgraph]]:
        """
        检索并重排序
        
        Args:
            query_embedding: 查询嵌入
            topk_retrieve: 初步检索数量
            topk_rerank: 重排序后数量
            subgraph_order: 子图阶数
            use_gnn_for_retrieve: 检索时是否使用 GNN
            use_gnn_for_rerank: 重排序时是否使用 GNN
            layer: GNN 层索引
        
        Returns:
            (重排序后的节点列表, 子图列表)
        """
        nodes = self.retrieve_topk_nodes_from_graph(
            query_embedding, topk_retrieve, use_gnn_for_retrieve, layer
        )
        
        reranked_nodes = self.re_ranking(
            query_embedding, nodes, topk_rerank, use_gnn_for_rerank, layer
        )
        
        subgraphs = []
        for node_idx in reranked_nodes:
            if subgraph_order == 1:
                subgraph = self.get_first_order_subgraph(node_idx)
            else:
                subgraph = self.get_second_order_subgraph(node_idx)
            subgraphs.append(subgraph)
        
        return reranked_nodes, subgraphs
    
    def subgraph_to_text(
        self,
        subgraph: Subgraph,
        max_length: int = 512
    ) -> str:
        """
        将子图序列化为文本
        
        Args:
            subgraph: 子图对象
            max_length: 最大文本长度
        
        Returns:
            文本描述
        """
        lines = []
        
        center_name = self.kg.node_names[subgraph.center_node]
        lines.append(f"中心实体: {center_name}")
        
        for src, dst, rel in subgraph.edges[:20]:
            src_name = self.kg.node_names[src] if src < len(self.kg.node_names) else str(src)
            dst_name = self.kg.node_names[dst] if dst < len(self.kg.node_names) else str(dst)
            rel_name = self.kg.relation_names[rel] if rel < len(self.kg.relation_names) else str(rel)
            lines.append(f"- {src_name} {rel_name} {dst_name}")
        
        text = "\n".join(lines)
        
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
