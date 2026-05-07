"""
GNN 编码器模块
实现多层图神经网络，用于知识图谱节点的多跳邻居信息聚合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple, List


class GNNEncoder(nn.Module):
    """
    基础 GNN 编码器
    使用 GCN 层进行节点嵌入学习，聚合多跳邻居信息
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        gnn_type: str = "gcn"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            
            if gnn_type == "gcn":
                self.convs.append(GCNConv(in_dim, out_dim))
            elif gnn_type == "gat":
                self.convs.append(GATConv(in_dim, out_dim, heads=4, concat=False))
            elif gnn_type == "sage":
                self.convs.append(SAGEConv(in_dim, out_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            if i < num_layers - 1:
                self.norms.append(nn.LayerNorm(out_dim))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges]
        
        Returns:
            节点嵌入 [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MultiLayerGNN(nn.Module):
    """
    多层 GNN 编码器
    分别输出不同层的嵌入，用于分层知识子图索引
    对应论文中的 layer2, layer3 嵌入
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GCNConv(in_dim, out_dim))
            self.norms.append(nn.LayerNorm(out_dim))
        
        self.dropout = dropout
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        前向传播，返回各层的嵌入
        
        Args:
            x: 节点特征
            edge_index: 边索引
        
        Returns:
            各层嵌入列表 [layer0_emb, layer1_emb, layer2_emb, ...]
        """
        embeddings = [x]
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            embeddings.append(x)
        
        return embeddings
    
    def get_layer_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        获取指定层的嵌入
        
        Args:
            x: 节点特征
            edge_index: 边索引
            layer_idx: 层索引 (1, 2, 3, ...)
        
        Returns:
            指定层的节点嵌入
        """
        assert 0 <= layer_idx <= self.num_layers
        
        for i in range(layer_idx):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class RelationalGNN(nn.Module):
    """
    关系感知 GNN 编码器
    考虑知识图谱中不同类型的关系
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_relations: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_relations = num_relations
        self.num_layers = num_layers
        
        from torch_geometric.nn import RGCNConv
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(RGCNConv(in_dim, out_dim, num_relations))
            self.norms.append(nn.LayerNorm(out_dim))
        
        self.dropout = dropout
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型 [num_edges]
        
        Returns:
            节点嵌入
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            x = self.norms[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SubgraphEncoder(nn.Module):
    """
    子图编码器
    将检索到的知识子图编码为固定维度的向量
    用于知识增强推荐生成
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.gnn = GNNEncoder(node_dim, hidden_dim, hidden_dim, num_layers=2)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码子图
        
        Args:
            x: 节点特征
            edge_index: 边索引
            node_mask: 节点掩码
        
        Returns:
            子图表征向量
        """
        node_emb = self.gnn(x, edge_index)
        
        node_emb = node_emb.unsqueeze(0)
        
        if node_mask is not None:
            attn_mask = ~node_mask.bool()
        else:
            attn_mask = None
        
        graph_emb, _ = self.attention(node_emb, node_emb, node_emb, key_padding_mask=attn_mask)
        
        graph_emb = graph_emb.mean(dim=1)
        
        return self.proj(graph_emb)
