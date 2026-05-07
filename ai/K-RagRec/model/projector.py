"""
知识投影器模块
将 GNN 编码的知识子图表征对齐到 LLM 语义空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class KnowledgeProjector(nn.Module):
    """
    知识投影器
    将 GNN 输出的知识嵌入投影到 LLM 的嵌入空间
    实现结构化知识与大语言模型的语义对齐
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        
        if activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "relu":
            layers.append(nn.ReLU())
        else:
            layers.append(nn.GELU())
        
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.projector = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: GNN 输出的知识嵌入 [batch_size, input_dim] 或 [num_nodes, input_dim]
        
        Returns:
            投影后的嵌入，对齐到 LLM 语义空间
        """
        return self.projector(x)


class SoftPromptProjector(nn.Module):
    """
    软提示投影器
    将知识嵌入转换为软提示 token，拼接到 LLM 输入
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        llm_hidden_dim: int,
        num_soft_tokens: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_soft_tokens = num_soft_tokens
        self.llm_hidden_dim = llm_hidden_dim
        
        self.knowledge_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, llm_hidden_dim * num_soft_tokens)
        )
        
        self.token_proj = nn.Sequential(
            nn.Linear(llm_hidden_dim, llm_hidden_dim),
            nn.LayerNorm(llm_hidden_dim)
        )
    
    def forward(self, knowledge_emb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            knowledge_emb: 知识嵌入 [batch_size, input_dim]
        
        Returns:
            软提示 token [batch_size, num_soft_tokens, llm_hidden_dim]
        """
        batch_size = knowledge_emb.size(0)
        
        soft_tokens = self.knowledge_proj(knowledge_emb)
        
        soft_tokens = soft_tokens.view(batch_size, self.num_soft_tokens, self.llm_hidden_dim)
        
        soft_tokens = self.token_proj(soft_tokens)
        
        return soft_tokens


class MultiModalProjector(nn.Module):
    """
    多模态投影器
    支持多种知识模态的融合投影
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        融合节点和边嵌入
        
        Args:
            node_emb: 节点嵌入
            edge_emb: 边嵌入
        
        Returns:
            融合后的嵌入
        """
        node_feat = self.node_proj(node_emb)
        edge_feat = self.edge_proj(edge_emb)
        
        if edge_feat.dim() == 2 and node_feat.dim() == 2:
            if edge_feat.size(0) != node_feat.size(0):
                edge_feat = edge_feat.mean(dim=0, keepdim=True).expand(node_feat.size(0), -1)
        
        combined = torch.cat([node_feat, edge_feat], dim=-1)
        
        return self.fusion(combined)
