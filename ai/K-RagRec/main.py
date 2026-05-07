"""
K-RagRec: Knowledge Graph Retrieval-Augmented Recommendation
ACL 2025

主入口文件
"""

import os
import argparse
import json
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from configs import Config, get_config
from model import GNNEncoder, MultiLayerGNN, KnowledgeProjector
from indexing import KnowledgeGraphIndexer, KnowledgeGraph
from retrieval import GraphRetrieval, AdaptiveRetriever, Subgraph
from generation import RecommendationGenerator, KnowledgeEnhancedGenerator
from utils import (
    MovieLensProcessor,
    create_sample_data,
    RecommendationEvaluator
)


class K_RagRec:
    """
    K-RagRec 完整框架
    
    整合所有模块，实现端到端的知识图谱检索增强推荐
    """
    
    def __init__(
        self,
        config: Config,
        device: str = "cuda"
    ):
        self.config = config
        self.device = device
        
        self.indexer: Optional[KnowledgeGraphIndexer] = None
        self.gnn_encoder: Optional[MultiLayerGNN] = None
        self.retrieval: Optional[GraphRetrieval] = None
        self.adaptive_retriever: Optional[AdaptiveRetriever] = None
        self.generator: Optional[RecommendationGenerator] = None
        self.projector: Optional[KnowledgeProjector] = None
        
        self.is_initialized = False
    
    def initialize(
        self,
        kg_dir: str,
        use_sample_data: bool = False
    ):
        """
        初始化所有组件
        
        Args:
            kg_dir: 知识图谱目录
            use_sample_data: 是否使用示例数据
        """
        print("=" * 60)
        print("Initializing K-RagRec...")
        print("=" * 60)
        
        print("\n[1/5] Loading Knowledge Graph...")
        self.indexer = KnowledgeGraphIndexer(
            encoder_model=self.config.model.encoder_model,
            device=self.device,
            embedding_dim=self.config.model.embedding_dim
        )
        
        if use_sample_data:
            create_sample_data(kg_dir)
        
        self.indexer.load_knowledge_graph(
            entity_file=os.path.join(kg_dir, "entities.txt"),
            relation_file=os.path.join(kg_dir, "relations.txt"),
            triple_file=os.path.join(kg_dir, "triples.txt")
        )
        print(f"  Loaded {self.indexer.kg.num_nodes} entities, {self.indexer.kg.num_edges} edges")
        
        print("\n[2/5] Encoding Entities...")
        self.indexer.encode_entities(batch_size=64)
        print(f"  Embedding shape: {self.indexer.node_embeddings.shape}")
        
        print("\n[3/5] Building GNN Encoder...")
        self.gnn_encoder = MultiLayerGNN(
            in_channels=self.config.model.embedding_dim,
            hidden_channels=self.config.model.gnn_hidden_dim,
            out_channels=self.config.model.gnn_output_dim,
            num_layers=self.config.model.gnn_num_layers,
            dropout=self.config.model.gnn_dropout
        ).to(self.device)
        
        self.indexer.build_layer_embeddings(self.gnn_encoder, layers=[1, 2, 3])
        print(f"  Built embeddings for layers: {list(self.indexer.layer_embeddings.keys())}")
        
        print("\n[4/5] Initializing Retrieval Components...")
        self.retrieval = GraphRetrieval(
            kg=self.indexer.kg,
            node_embeddings=self.indexer.node_embeddings,
            gnn_embeddings=self.indexer.layer_embeddings,
            device=self.device
        )
        print("  GraphRetrieval initialized")
        
        self.adaptive_retriever = AdaptiveRetriever(
            popularity_threshold=self.config.retrieval.popularity_threshold,
            use_percentile=True
        )
        print("  AdaptiveRetriever initialized")
        
        print("\n[5/5] Loading LLM Generator...")
        self.generator = RecommendationGenerator(
            model_name=self.config.model.llm_model_name,
            device=self.device,
            max_length=self.config.model.llm_max_length,
            temperature=self.config.model.llm_temperature
        )
        print(f"  Loaded: {self.config.model.llm_model_name}")
        
        self.projector = KnowledgeProjector(
            input_dim=self.config.model.gnn_output_dim,
            hidden_dim=self.config.model.projector_hidden_dim,
            output_dim=self.config.model.projector_output_dim
        ).to(self.device)
        
        self.is_initialized = True
        print("\n" + "=" * 60)
        print("K-RagRec Initialization Complete!")
        print("=" * 60)
    
    def recommend(
        self,
        user_history: List[str],
        num_recommendations: int = 5,
        use_adaptive: bool = True
    ) -> str:
        """
        生成推荐
        
        Args:
            user_history: 用户历史物品名称列表
            num_recommendations: 推荐数量
            use_adaptive: 是否使用自适应检索
        
        Returns:
            推荐文本
        """
        if not self.is_initialized:
            raise RuntimeError("请先调用 initialize() 初始化框架")
        
        query_text = " ".join(user_history[-5:])
        query_embedding = self.indexer.encoder.encode(
            query_text,
            convert_to_tensor=True
        ).to(self.device)
        
        nodes, subgraphs = self.retrieval.retrieve_and_rerank(
            query_embedding=query_embedding,
            topk_retrieve=self.config.retrieval.topk_nodes,
            topk_rerank=self.config.retrieval.topk_rerank,
            subgraph_order=2,
            use_gnn_for_retrieve=False,
            use_gnn_for_rerank=True,
            layer=2
        )
        
        knowledge_texts = []
        for subgraph in subgraphs:
            text = self.retrieval.subgraph_to_text(subgraph)
            knowledge_texts.append(text)
        
        recommendation = self.generator.generate_recommendations(
            user_history=user_history,
            retrieved_knowledge=knowledge_texts,
            num_recommendations=num_recommendations,
            use_chat_format=True
        )
        
        return recommendation
    
    def save(self, save_dir: str):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.indexer:
            self.indexer.save_index(save_dir, prefix="kragrec_")
        
        if self.gnn_encoder:
            torch.save(
                self.gnn_encoder.state_dict(),
                os.path.join(save_dir, "gnn_encoder.pt")
            )
        
        if self.projector:
            torch.save(
                self.projector.state_dict(),
                os.path.join(save_dir, "projector.pt")
            )
        
        config_dict = {
            "model": vars(self.config.model),
            "retrieval": vars(self.config.retrieval),
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Model saved to {save_dir}")
    
    def load(self, load_dir: str):
        """加载模型"""
        if self.indexer:
            self.indexer.load_index(load_dir, prefix="kragrec_")
        
        if self.gnn_encoder:
            self.gnn_encoder.load_state_dict(
                torch.load(os.path.join(load_dir, "gnn_encoder.pt"))
            )
        
        if self.projector:
            self.projector.load_state_dict(
                torch.load(os.path.join(load_dir, "projector.pt"))
            )
        
        print(f"Model loaded from {load_dir}")


def main():
    parser = argparse.ArgumentParser(description="K-RagRec: Knowledge Graph Retrieval-Augmented Recommendation")
    
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "train", "eval"],
                        help="运行模式")
    parser.add_argument("--kg_dir", type=str, default="data/kg",
                        help="知识图谱目录")
    parser.add_argument("--data_dir", type=str, default="data/ml-1m",
                        help="MovieLens 数据目录")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    parser.add_argument("--use_sample_data", action="store_true",
                        help="使用示例数据")
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="LLM 模型名称")
    parser.add_argument("--num_recommendations", type=int, default=5,
                        help="推荐数量")
    
    args = parser.parse_args()
    
    config = get_config()
    config.model.llm_model_name = args.llm_model
    
    print("\n" + "=" * 60)
    print("K-RagRec: Knowledge Graph Retrieval-Augmented Recommendation")
    print("ACL 2025")
    print("=" * 60)
    
    model = K_RagRec(config, device=args.device)
    
    kg_dir = os.path.join(os.path.dirname(__file__), args.kg_dir)
    model.initialize(kg_dir, use_sample_data=args.use_sample_data)
    
    if args.mode == "demo":
        print("\n" + "=" * 60)
        print("Demo Mode: Generating Recommendations")
        print("=" * 60)
        
        user_history = [
            "Toy Story (1995)",
            "Jumanji (1995)",
            "The Matrix (1999)",
            "Inception (2010)"
        ]
        
        print(f"\nUser History: {user_history}")
        print("\nGenerating recommendations...\n")
        
        recommendation = model.recommend(
            user_history=user_history,
            num_recommendations=args.num_recommendations
        )
        
        print("Recommendations:")
        print("-" * 40)
        print(recommendation)
        print("-" * 40)
    
    elif args.mode == "train":
        print("\n[Training Mode]")
        print("Training pipeline will be implemented...")
    
    elif args.mode == "eval":
        print("\n[Evaluation Mode]")
        print("Evaluation pipeline will be implemented...")


if __name__ == "__main__":
    main()
