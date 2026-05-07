"""
K-RagRec: Knowledge Graph Retrieval-Augmented Recommendation
ACL 2025
"""

from .model.gnn_encoder import GNNEncoder, MultiLayerGNN
from .model.projector import KnowledgeProjector
from .indexing.index_KG import KnowledgeGraphIndexer
from .retrieval.retrieve import GraphRetrieval
from .retrieval.retriever import AdaptiveRetriever
from .generation.generator import RecommendationGenerator

__version__ = "1.0.0"
__all__ = [
    "GNNEncoder",
    "MultiLayerGNN",
    "KnowledgeProjector",
    "KnowledgeGraphIndexer",
    "GraphRetrieval",
    "AdaptiveRetriever",
    "RecommendationGenerator",
]
