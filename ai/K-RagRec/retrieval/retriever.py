"""
自适应检索策略模块
基于物品流行度动态决定是否需要检索知识图谱
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm


@dataclass
class ItemStatistics:
    """物品统计信息"""
    item_id: int
    popularity: float
    interaction_count: int
    need_retrieval: bool


class AdaptiveRetriever:
    """
    自适应检索器
    
    对应论文模块2：自适应检索策略
    - 基于物品流行度动态决定是否需要检索
    - 显著提升检索效率，降低计算开销
    - 阈值 p 可调节，实现效率与性能的平衡
    
    核心思想：
    热门物品的知识大模型已经知道，无需检索
    冷门物品需要从知识图谱补充信息
    """
    
    def __init__(
        self,
        popularity_threshold: float = 0.1,
        min_interactions: int = 5,
        use_percentile: bool = True
    ):
        """
        初始化自适应检索器
        
        Args:
            popularity_threshold: 流行度阈值
            min_interactions: 最小交互次数
            use_percentile: 是否使用百分位数作为阈值
        """
        self.popularity_threshold = popularity_threshold
        self.min_interactions = min_interactions
        self.use_percentile = use_percentile
        
        self.item_stats: Dict[int, ItemStatistics] = {}
        self.popularity_scores: Optional[np.ndarray] = None
    
    def compute_popularity(
        self,
        interactions: List[Tuple[int, int]],
        num_items: int
    ) -> Dict[int, float]:
        """
        计算物品流行度
        
        Args:
            interactions: 用户-物品交互列表 [(user_id, item_id), ...]
            num_items: 物品总数
        
        Returns:
            物品流行度字典 {item_id: popularity}
        """
        item_counts = np.zeros(num_items)
        
        for _, item_id in interactions:
            if 0 <= item_id < num_items:
                item_counts[item_id] += 1
        
        total_interactions = len(interactions)
        if total_interactions > 0:
            popularity = item_counts / total_interactions
        else:
            popularity = item_counts
        
        self.popularity_scores = popularity
        
        return {i: popularity[i] for i in range(num_items)}
    
    def build_item_statistics(
        self,
        interactions: List[Tuple[int, int]],
        num_items: int
    ):
        """
        构建物品统计信息
        
        Args:
            interactions: 用户-物品交互列表
            num_items: 物品总数
        """
        popularity_dict = self.compute_popularity(interactions, num_items)
        
        item_counts = {}
        for _, item_id in interactions:
            item_counts[item_id] = item_counts.get(item_id, 0) + 1
        
        if self.use_percentile:
            threshold = np.percentile(
                list(popularity_dict.values()),
                self.popularity_threshold * 100
            )
        else:
            threshold = self.popularity_threshold
        
        for item_id in range(num_items):
            pop = popularity_dict.get(item_id, 0.0)
            count = item_counts.get(item_id, 0)
            
            need_retrieval = pop < threshold or count < self.min_interactions
            
            self.item_stats[item_id] = ItemStatistics(
                item_id=item_id,
                popularity=pop,
                interaction_count=count,
                need_retrieval=need_retrieval
            )
    
    def should_retrieve(self, item_id: int) -> bool:
        """
        判断是否需要检索该物品的知识
        
        Args:
            item_id: 物品 ID
        
        Returns:
            是否需要检索
        """
        if item_id not in self.item_stats:
            return True
        
        return self.item_stats[item_id].need_retrieval
    
    def filter_items_for_retrieval(
        self,
        item_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        过滤物品，区分需要检索和不需要检索的
        
        Args:
            item_ids: 物品 ID 列表
        
        Returns:
            (需要检索的物品列表, 不需要检索的物品列表)
        """
        need_retrieval = []
        no_retrieval = []
        
        for item_id in item_ids:
            if self.should_retrieve(item_id):
                need_retrieval.append(item_id)
            else:
                no_retrieval.append(item_id)
        
        return need_retrieval, no_retrieval
    
    def get_retrieval_efficiency(self) -> float:
        """
        获取检索效率提升比例
        
        Returns:
            跳过检索的物品比例
        """
        if not self.item_stats:
            return 0.0
        
        skipped = sum(1 for stat in self.item_stats.values() if not stat.need_retrieval)
        return skipped / len(self.item_stats)
    
    def update_threshold(self, new_threshold: float):
        """
        更新流行度阈值
        
        Args:
            new_threshold: 新的阈值
        """
        self.popularity_threshold = new_threshold
        
        if self.popularity_scores is not None:
            if self.use_percentile:
                threshold = np.percentile(
                    self.popularity_scores,
                    new_threshold * 100
                )
            else:
                threshold = new_threshold
            
            for item_id, stat in self.item_stats.items():
                stat.need_retrieval = (
                    stat.popularity < threshold or 
                    stat.interaction_count < self.min_interactions
                )
    
    def get_statistics_summary(self) -> Dict:
        """
        获取统计信息摘要
        
        Returns:
            统计信息字典
        """
        if not self.item_stats:
            return {}
        
        popularities = [s.popularity for s in self.item_stats.values()]
        counts = [s.interaction_count for s in self.item_stats.values()]
        
        return {
            "total_items": len(self.item_stats),
            "items_need_retrieval": sum(1 for s in self.item_stats.values() if s.need_retrieval),
            "items_skip_retrieval": sum(1 for s in self.item_stats.values() if not s.need_retrieval),
            "retrieval_efficiency": self.get_retrieval_efficiency(),
            "popularity_mean": np.mean(popularities),
            "popularity_std": np.std(popularities),
            "interaction_mean": np.mean(counts),
            "interaction_std": np.std(counts),
        }


class BatchAdaptiveRetriever(AdaptiveRetriever):
    """
    批量自适应检索器
    支持批量处理，进一步提升效率
    """
    
    def __init__(
        self,
        popularity_threshold: float = 0.1,
        min_interactions: int = 5,
        use_percentile: bool = True,
        batch_size: int = 256
    ):
        super().__init__(popularity_threshold, min_interactions, use_percentile)
        self.batch_size = batch_size
    
    def batch_should_retrieve(self, item_ids: List[int]) -> List[bool]:
        """
        批量判断是否需要检索
        
        Args:
            item_ids: 物品 ID 列表
        
        Returns:
            布尔列表
        """
        return [self.should_retrieve(item_id) for item_id in item_ids]
    
    def batch_filter(
        self,
        item_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        批量过滤物品
        
        Args:
            item_ids: 物品 ID 列表
        
        Returns:
            (需要检索的物品列表, 不需要检索的物品列表)
        """
        return self.filter_items_for_retrieval(item_ids)
