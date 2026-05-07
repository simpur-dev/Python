"""
K-RagRec 配置文件
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """模型配置"""
    gnn_hidden_dim: int = 128
    gnn_output_dim: int = 256
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.1
    
    projector_hidden_dim: int = 512
    projector_output_dim: int = 4096
    
    llm_model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    llm_max_length: int = 2048
    llm_temperature: float = 0.7
    
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384


@dataclass
class RetrievalConfig:
    """检索配置"""
    topk_nodes: int = 10
    topk_rerank: int = 5
    popularity_threshold: float = 0.1
    use_adaptive_retrieval: bool = True
    
    faiss_nprobe: int = 10
    faiss_nlist: int = 100


@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str = "ml-1m"
    data_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(__file__), "..", "data"))
    
    kg_entity_file: str = "entities.txt"
    kg_relation_file: str = "relations.txt"
    kg_triple_file: str = "triples.txt"
    
    user_item_file: str = "ratings.dat"
    item_meta_file: str = "movies.dat"
    
    max_history_length: int = 20
    min_interaction_count: int = 5


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 100
    
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    save_steps: int = 500
    eval_steps: int = 100
    log_steps: int = 10
    
    output_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(__file__), "..", "checkpoints"))


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    seed: int = 42
    device: str = "cuda"
    debug: bool = False


def get_config() -> Config:
    """获取默认配置"""
    return Config()
