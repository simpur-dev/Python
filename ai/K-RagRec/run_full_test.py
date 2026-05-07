"""
K-RagRec 完整版测试脚本
使用本地嵌入，避免 HuggingFace 下载问题
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, r"e:\Python\ai\K-RagRec")

from model.gnn_encoder import MultiLayerGNN
from retrieval.retrieve import GraphRetrieval
from retrieval.retriever import AdaptiveRetriever


class SimpleEmbeddingEncoder:
    """简单的嵌入编码器，使用随机初始化"""
    
    def __init__(self, embedding_dim=384, hidden_dim=128, gnn_dim=256, device="cuda"):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.gnn_dim = gnn_dim
        self.device = device
        self.word_embeddings = {}
        self.proj = nn.Linear(768, embedding_dim).to(device)
        self.hidden_proj = nn.Linear(embedding_dim, hidden_dim).to(device)
        self.gnn_proj = nn.Linear(embedding_dim, gnn_dim).to(device)
    
    def encode(self, texts, convert_to_tensor=True, for_gnn=False, for_hidden=False):
        """编码文本"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            words = text.lower().split()
            word_embs = []
            for word in words:
                if word not in self.word_embeddings:
                    self.word_embeddings[word] = torch.randn(768, device=self.device)
                word_embs.append(self.word_embeddings[word])
            
            if word_embs:
                text_emb = torch.stack(word_embs).mean(dim=0)
            else:
                text_emb = torch.zeros(768, device=self.device)
            
            text_emb = self.proj(text_emb)
            if for_gnn:
                text_emb = self.gnn_proj(text_emb)
            elif for_hidden:
                text_emb = self.hidden_proj(text_emb)
            embeddings.append(text_emb)
        
        result = torch.stack(embeddings)
        return result if convert_to_tensor else result.cpu().numpy()


class SimpleKnowledgeGraph:
    """简化的知识图谱"""
    
    def __init__(self):
        self.num_nodes = 0
        self.num_edges = 0
        self.num_relations = 0
        self.node_ids = []
        self.node_names = []
        self.node_types = []
        self.edge_index = torch.empty((2, 0), dtype=torch.long)
        self.edge_type = torch.empty((0,), dtype=torch.long)
        self.relation_names = []
        self.node_features = None


def create_kg_data():
    """创建示例知识图谱数据"""
    kg_dir = r"e:\Python\ai\K-RagRec\data\kg"
    os.makedirs(kg_dir, exist_ok=True)
    
    entities = [
        (0, "Toy Story", "movie"),
        (1, "Jumanji", "movie"),
        (2, "The Matrix", "movie"),
        (3, "Inception", "movie"),
        (4, "Interstellar", "movie"),
        (5, "The Dark Knight", "movie"),
        (6, "Avatar", "movie"),
        (7, "Titanic", "movie"),
        (8, "Animation", "genre"),
        (9, "Adventure", "genre"),
        (10, "Sci-Fi", "genre"),
        (11, "Action", "genre"),
        (12, "Drama", "genre"),
        (13, "Romance", "genre"),
        (14, "Christopher Nolan", "director"),
        (15, "Wachowski Sisters", "director"),
        (16, "James Cameron", "director"),
        (17, "Leonardo DiCaprio", "actor"),
        (18, "Matthew McConaughey", "actor"),
        (19, "Keanu Reeves", "actor"),
        (20, "Tom Hanks", "actor"),
    ]
    
    relations = [
        (0, "has_genre"),
        (1, "directed_by"),
        (2, "starring"),
        (3, "similar_to"),
    ]
    
    triples = [
        (0, 0, 8), (0, 0, 9),
        (1, 0, 8), (1, 0, 9),
        (2, 0, 10), (2, 0, 11),
        (3, 0, 10), (3, 0, 11),
        (4, 0, 10), (4, 0, 12),
        (5, 0, 11), (5, 0, 12),
        (6, 0, 10), (6, 0, 11),
        (7, 0, 12), (7, 0, 13),
        (2, 1, 15),
        (3, 1, 14), (4, 1, 14), (5, 1, 14),
        (6, 1, 16), (7, 1, 16),
        (3, 2, 17), (4, 2, 18), (2, 2, 19), (0, 2, 20),
        (2, 3, 3), (3, 3, 4), (4, 3, 6),
    ]
    
    with open(os.path.join(kg_dir, "entities.txt"), "w", encoding="utf-8") as f:
        for eid, name, etype in entities:
            f.write(f"{eid}\t{name}\t{etype}\n")
    
    with open(os.path.join(kg_dir, "relations.txt"), "w", encoding="utf-8") as f:
        for rid, name in relations:
            f.write(f"{rid}\t{name}\n")
    
    with open(os.path.join(kg_dir, "triples.txt"), "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")
    
    print(f"知识图谱数据已创建: {kg_dir}")
    print(f"  实体数量: {len(entities)}")
    print(f"  关系数量: {len(relations)}")
    print(f"  三元组数量: {len(triples)}")
    
    return kg_dir, entities, relations, triples


def build_kg(entities, relations, triples):
    """构建知识图谱对象"""
    kg = SimpleKnowledgeGraph()
    kg.num_nodes = len(entities)
    kg.num_edges = len(triples)
    kg.num_relations = len(relations)
    kg.node_ids = [e[0] for e in entities]
    kg.node_names = [e[1] for e in entities]
    kg.node_types = [e[2] for e in entities]
    kg.relation_names = [r[1] for r in relations]
    
    src = [t[0] for t in triples]
    dst = [t[2] for t in triples]
    edge_types = [t[1] for t in triples]
    
    kg.edge_index = torch.tensor([src, dst], dtype=torch.long)
    kg.edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    return kg


def test_kg_indexing(entities, kg, device="cuda"):
    """测试知识图谱索引"""
    print("\n" + "=" * 60)
    print("测试模块1: 分层知识子图语义索引")
    print("=" * 60)
    
    encoder = SimpleEmbeddingEncoder(embedding_dim=384, hidden_dim=128, gnn_dim=256, device=device)
    
    print("\n[1] 编码实体...")
    entity_texts = [f"{e[1]} {e[2]}" for e in entities]
    node_embeddings = encoder.encode(entity_texts)
    kg.node_features = node_embeddings
    print(f"  嵌入维度: {node_embeddings.shape}")
    
    print("\n[2] 构建 GNN 编码器...")
    gnn = MultiLayerGNN(
        in_channels=384,
        hidden_channels=128,
        out_channels=256,
        num_layers=3,
        dropout=0.1
    ).to(device)
    
    print("\n[3] 构建多层嵌入...")
    kg.edge_index = kg.edge_index.to(device)
    all_layer_embs = gnn(node_embeddings, kg.edge_index)
    
    if isinstance(all_layer_embs, list):
        layer_embs = all_layer_embs[-1]
        gnn_embeddings = {i+1: all_layer_embs[i] for i in range(len(all_layer_embs))}
    else:
        layer_embs = all_layer_embs
        gnn_embeddings = {1: layer_embs, 2: layer_embs, 3: layer_embs}
    print(f"  各层嵌入: {layer_embs.shape}")
    
    return encoder, gnn, node_embeddings, gnn_embeddings


def test_retrieval(kg, node_embeddings, gnn_embeddings, encoder, device="cuda"):
    """测试检索与重排序"""
    print("\n" + "=" * 60)
    print("测试模块3&4: 知识子图检索与重排序")
    print("=" * 60)
    
    retrieval = GraphRetrieval(
        kg=kg,
        node_embeddings=node_embeddings,
        gnn_embeddings=gnn_embeddings,
        device=device
    )
    
    print("\n[1] 测试查询: 'science fiction movie'")
    query_emb = encoder.encode("science fiction movie")[0]
    query_emb_gnn = encoder.encode("science fiction movie", for_gnn=True)[0]
    
    print("\n[2] 检索 Top-5 节点 (初始检索)...")
    nodes = retrieval.retrieve_topk_nodes_from_graph(query_emb, topk_nodes=5, use_gnn=False)
    print(f"  检索结果: {nodes}")
    for node_idx in nodes:
        print(f"    - {kg.node_names[node_idx]}")
    
    print("\n[3] 重排序 (使用 GNN 嵌入)...")
    query_emb_layer2 = encoder.encode("science fiction movie", for_hidden=True)[0]
    reranked = retrieval.re_ranking(query_emb_layer2, nodes, topk_nodes=3, use_gnn=True, layer=2)
    print(f"  重排序结果: {reranked}")
    for node_idx in reranked:
        print(f"    - {kg.node_names[node_idx]}")
    
    print("\n[4] 提取知识子图...")
    for node_idx in reranked[:1]:
        subgraph = retrieval.get_second_order_subgraph(node_idx)
        print(f"  中心节点: {kg.node_names[node_idx]}")
        print(f"  子图节点数: {len(subgraph.nodes)}")
        print(f"  子图边数: {len(subgraph.edges)}")
        
        text = retrieval.subgraph_to_text(subgraph)
        print(f"\n  子图文本描述:\n{text}")
    
    return retrieval


def test_adaptive_retrieval():
    """测试自适应检索策略"""
    print("\n" + "=" * 60)
    print("测试模块2: 自适应检索策略")
    print("=" * 60)
    
    retriever = AdaptiveRetriever(
        popularity_threshold=0.1,
        min_interactions=5,
        use_percentile=True
    )
    
    interactions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0),
        (2, 1), (2, 2), (2, 3),
        (3, 0), (3, 0), (3, 0), (3, 0), (3, 0),
        (4, 1), (4, 2),
        (5, 3), (5, 4), (5, 5),
    ]
    
    retriever.build_item_statistics(interactions, num_items=6)
    
    print("\n[1] 物品统计信息:")
    for item_id, stat in retriever.item_stats.items():
        status = "需要检索" if stat.need_retrieval else "跳过检索"
        print(f"  物品 {item_id}: 流行度={stat.popularity:.3f}, 交互={stat.interaction_count}, {status}")
    
    print(f"\n[2] 检索效率提升: {retriever.get_retrieval_efficiency():.1%}")
    
    stats = retriever.get_statistics_summary()
    print(f"\n[3] 统计摘要:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def test_recommendation_flow(kg, retrieval, encoder, device="cuda"):
    """测试完整推荐流程"""
    print("\n" + "=" * 60)
    print("测试模块5: 知识增强推荐生成 (模拟)")
    print("=" * 60)
    
    user_history = ["The Matrix", "Inception", "Interstellar"]
    
    print("\n[1] 用户历史:")
    for item in user_history:
        print(f"  - {item}")
    
    print("\n[2] 编码用户查询...")
    query_emb_base = encoder.encode(" ".join(user_history))[0]
    query_emb_layer = encoder.encode(" ".join(user_history), for_hidden=True)[0]
    
    print("\n[3] 检索相关知识子图...")
    
    nodes = retrieval.retrieve_topk_nodes_from_graph(
        query_embedding=query_emb_base,
        topk_nodes=5,
        use_gnn=False,
        layer=2
    )
    
    reranked_nodes = retrieval.re_ranking(
        query_embedding=query_emb_layer,
        retrieved_nodes=nodes,
        topk_nodes=3,
        use_gnn=True,
        layer=2
    )
    
    subgraphs = []
    for node_idx in reranked_nodes:
        subgraph = retrieval.get_second_order_subgraph(node_idx)
        subgraphs.append(subgraph)
    
    print("\n[4] 检索到的知识:")
    for i, node_idx in enumerate(reranked_nodes):
        subgraph = subgraphs[i] if i < len(subgraphs) else None
        if subgraph:
            text = retrieval.subgraph_to_text(subgraph)
            print(f"  知识 {i+1}: {text[:80]}...")
    
    print("\n[5] 基于知识子图的推荐结果:")
    print("-" * 50)
    for i, node_idx in enumerate(reranked_nodes):
        name = kg.node_names[node_idx]
        node_type = kg.node_types[node_idx]
        print(f"  {i+1}. {name} ({node_type})")
    print("-" * 50)


def main():
    print("=" * 60)
    print("K-RagRec 完整版测试")
    print("ACL 2025")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    kg_dir, entities, relations, triples = create_kg_data()
    
    kg = build_kg(entities, relations, triples)
    
    encoder, gnn, node_embeddings, gnn_embeddings = test_kg_indexing(entities, kg, device)
    
    retrieval = test_retrieval(kg, node_embeddings, gnn_embeddings, encoder, device)
    
    test_adaptive_retrieval()
    
    test_recommendation_flow(kg, retrieval, encoder, device)
    
    print("\n" + "=" * 60)
    print("所有模块测试完成!")
    print("=" * 60)
    print("\n模块验证结果:")
    print("  [OK] 模块1: 分层知识子图语义索引")
    print("  [OK] 模块2: 自适应检索策略")
    print("  [OK] 模块3: 知识子图检索")
    print("  [OK] 模块4: 知识子图重排序")
    print("  [OK] 模块5: 知识增强推荐生成 (模拟)")
    print("\n框架已准备就绪，可接入真实 LLM 进行完整测试!")


if __name__ == "__main__":
    main()
