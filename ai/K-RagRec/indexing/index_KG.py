"""
知识图谱索引模块
实现分层知识子图语义索引，构建知识向量数据库
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer


@dataclass
class KnowledgeGraph:
    """知识图谱数据结构"""
    num_nodes: int
    num_edges: int
    num_relations: int
    
    node_ids: List[int]
    node_names: List[str]
    node_types: List[str]
    
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    
    relation_names: List[str]
    
    node_features: Optional[torch.Tensor] = None
    
    def to(self, device: str) -> "KnowledgeGraph":
        """移动到指定设备"""
        self.edge_index = self.edge_index.to(device)
        self.edge_type = self.edge_type.to(device)
        if self.node_features is not None:
            self.node_features = self.node_features.to(device)
        return self


class KnowledgeGraphIndexer:
    """
    知识图谱索引器
    实现分层知识子图语义索引
    
    对应论文模块1：分层知识子图语义索引
    - 使用预训练语言模型对实体和关系进行语义编码
    - 引入 GNN 进行多跳邻居信息聚合
    - 构建粗粒度和细粒度的知识向量索引
    """
    
    def __init__(
        self,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        embedding_dim: int = 384
    ):
        self.device = device
        self.embedding_dim = embedding_dim
        
        self.encoder = SentenceTransformer(encoder_model, device=device)
        
        self.kg: Optional[KnowledgeGraph] = None
        self.node_embeddings: Optional[torch.Tensor] = None
        self.layer_embeddings: Dict[int, torch.Tensor] = {}
        
        self.faiss_index = None
        self.layer_faiss_indices: Dict[int, faiss.Index] = {}
    
    def load_knowledge_graph(
        self,
        entity_file: str,
        relation_file: str,
        triple_file: str
    ) -> KnowledgeGraph:
        """
        加载知识图谱
        
        Args:
            entity_file: 实体文件路径
            relation_file: 关系文件路径
            triple_file: 三元组文件路径
        
        Returns:
            KnowledgeGraph 对象
        """
        node_ids = []
        node_names = []
        node_types = []
        
        with open(entity_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    node_ids.append(int(parts[0]))
                    node_names.append(parts[1])
                    node_types.append(parts[2] if len(parts) > 2 else "entity")
        
        relation_names = []
        with open(relation_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 1:
                    relation_names.append(parts[0])
        
        edges_src = []
        edges_dst = []
        edge_types = []
        
        with open(triple_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                    edges_src.append(h)
                    edges_dst.append(t)
                    edge_types.append(r)
        
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        self.kg = KnowledgeGraph(
            num_nodes=len(node_ids),
            num_edges=len(edges_src),
            num_relations=len(relation_names),
            node_ids=node_ids,
            node_names=node_names,
            node_types=node_types,
            edge_index=edge_index,
            edge_type=edge_type,
            relation_names=relation_names
        )
        
        return self.kg
    
    def encode_entities(
        self,
        use_descriptions: bool = True,
        batch_size: int = 256
    ) -> torch.Tensor:
        """
        对实体进行语义编码
        
        Args:
            use_descriptions: 是否使用实体描述
            batch_size: 批处理大小
        
        Returns:
            实体嵌入矩阵 [num_nodes, embedding_dim]
        """
        if self.kg is None:
            raise ValueError("请先加载知识图谱")
        
        texts = []
        for i, name in enumerate(self.kg.node_names):
            if use_descriptions and i < len(self.kg.node_types):
                text = f"{name} ({self.kg.node_types[i]})"
            else:
                text = name
            texts.append(text)
        
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True
        )
        
        self.node_embeddings = embeddings.to(self.device)
        self.kg.node_features = self.node_embeddings
        
        return self.node_embeddings
    
    def build_layer_embeddings(
        self,
        gnn_encoder: nn.Module,
        layers: List[int] = [1, 2, 3]
    ) -> Dict[int, torch.Tensor]:
        """
        构建多层 GNN 嵌入
        
        Args:
            gnn_encoder: GNN 编码器
            layers: 需要保存的层列表
        
        Returns:
            各层嵌入字典 {layer_idx: embedding}
        """
        if self.kg is None or self.node_embeddings is None:
            raise ValueError("请先加载知识图谱并编码实体")
        
        gnn_encoder = gnn_encoder.to(self.device)
        gnn_encoder.eval()
        
        with torch.no_grad():
            all_embeddings = gnn_encoder(
                self.node_embeddings,
                self.kg.edge_index.to(self.device)
            )
        
        if isinstance(all_embeddings, list):
            for layer_idx in layers:
                if layer_idx < len(all_embeddings):
                    self.layer_embeddings[layer_idx] = all_embeddings[layer_idx]
        else:
            self.layer_embeddings[1] = all_embeddings
        
        return self.layer_embeddings
    
    def build_faiss_index(
        self,
        use_gpu: bool = True,
        nlist: int = 100
    ) -> faiss.Index:
        """
        构建 FAISS 向量索引
        
        Args:
            use_gpu: 是否使用 GPU
            nlist: 聚类中心数量
        
        Returns:
            FAISS 索引
        """
        if not FAISS_AVAILABLE:
            raise ImportError("请安装 faiss: pip install faiss-cpu 或 faiss-gpu")
        
        if self.node_embeddings is None:
            raise ValueError("请先编码实体")
        
        embeddings_np = self.node_embeddings.cpu().numpy().astype("float32")
        
        d = embeddings_np.shape[1]
        
        quantizer = faiss.IndexFlatL2(d)
        self.faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist)
        
        self.faiss_index.train(embeddings_np)
        self.faiss_index.add(embeddings_np)
        
        if use_gpu and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        
        return self.faiss_index
    
    def build_layer_faiss_indices(
        self,
        use_gpu: bool = True,
        nlist: int = 100
    ) -> Dict[int, faiss.Index]:
        """
        为各层嵌入构建 FAISS 索引
        
        Args:
            use_gpu: 是否使用 GPU
            nlist: 聚类中心数量
        
        Returns:
            各层 FAISS 索引字典
        """
        if not FAISS_AVAILABLE:
            raise ImportError("请安装 faiss")
        
        for layer_idx, embeddings in self.layer_embeddings.items():
            embeddings_np = embeddings.cpu().numpy().astype("float32")
            
            d = embeddings_np.shape[1]
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            
            index.train(embeddings_np)
            index.add(embeddings_np)
            
            if use_gpu and torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            
            self.layer_faiss_indices[layer_idx] = index
        
        return self.layer_faiss_indices
    
    def save_index(
        self,
        save_dir: str,
        prefix: str = ""
    ):
        """
        保存索引到文件
        
        Args:
            save_dir: 保存目录
            prefix: 文件名前缀
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if self.node_embeddings is not None:
            torch.save(
                self.node_embeddings.cpu(),
                os.path.join(save_dir, f"{prefix}0.pt")
            )
        
        for layer_idx, embeddings in self.layer_embeddings.items():
            torch.save(
                embeddings.cpu(),
                os.path.join(save_dir, f"{prefix}layer{layer_idx}_embeddings.pt")
            )
        
        if self.kg is not None:
            with open(os.path.join(save_dir, f"{prefix}kg.pkl"), "wb") as f:
                pickle.dump(self.kg, f)
        
        if self.faiss_index is not None and FAISS_AVAILABLE:
            if hasattr(self.faiss_index, "getDevice") and self.faiss_index.getDevice() >= 0:
                cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
            else:
                cpu_index = self.faiss_index
            faiss.write_index(cpu_index, os.path.join(save_dir, f"{prefix}faiss_index.bin"))
    
    def load_index(
        self,
        save_dir: str,
        prefix: str = ""
    ):
        """
        从文件加载索引
        
        Args:
            save_dir: 保存目录
            prefix: 文件名前缀
        """
        if os.path.exists(os.path.join(save_dir, f"{prefix}0.pt")):
            self.node_embeddings = torch.load(
                os.path.join(save_dir, f"{prefix}0.pt"),
                map_location=self.device
            )
        
        layer = 1
        while os.path.exists(os.path.join(save_dir, f"{prefix}layer{layer}_embeddings.pt")):
            self.layer_embeddings[layer] = torch.load(
                os.path.join(save_dir, f"{prefix}layer{layer}_embeddings.pt"),
                map_location=self.device
            )
            layer += 1
        
        if os.path.exists(os.path.join(save_dir, f"{prefix}kg.pkl")):
            with open(os.path.join(save_dir, f"{prefix}kg.pkl"), "rb") as f:
                self.kg = pickle.load(f)
        
        if os.path.exists(os.path.join(save_dir, f"{prefix}faiss_index.bin")) and FAISS_AVAILABLE:
            self.faiss_index = faiss.read_index(
                os.path.join(save_dir, f"{prefix}faiss_index.bin")
            )
    
    def search(
        self,
        query_embedding: torch.Tensor,
        k: int = 10,
        use_layer: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        搜索最相似的节点
        
        Args:
            query_embedding: 查询嵌入
            k: 返回数量
            use_layer: 使用指定层的嵌入
        
        Returns:
            (相似度分数, 节点索引)
        """
        if use_layer is not None and use_layer in self.layer_faiss_indices:
            index = self.layer_faiss_indices[use_layer]
        elif self.faiss_index is not None:
            index = self.faiss_index
        else:
            raise ValueError("请先构建 FAISS 索引")
        
        query_np = query_embedding.cpu().numpy().astype("float32")
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)
        
        distances, indices = index.search(query_np, k)
        
        return torch.tensor(distances), torch.tensor(indices)
