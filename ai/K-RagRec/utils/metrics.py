"""
评估指标模块
"""

from typing import Dict, List, Tuple
from collections import Counter

import numpy as np


def hit_rate(
    predictions: List[List[int]],
    ground_truth: List[int],
    k: int = 10
) -> float:
    """
    计算 Hit Rate@K
    
    Args:
        predictions: 预测列表，每个元素是 top-k 预测物品列表
        ground_truth: 真实物品列表
        k: 截断位置
    
    Returns:
        Hit Rate
    """
    hits = 0
    for pred, gt in zip(predictions, ground_truth):
        if gt in pred[:k]:
            hits += 1
    return hits / len(ground_truth) if ground_truth else 0.0


def ndcg_at_k(
    predictions: List[List[int]],
    ground_truth: List[int],
    k: int = 10
) -> float:
    """
    计算 NDCG@K
    
    Args:
        predictions: 预测列表
        ground_truth: 真实物品列表
        k: 截断位置
    
    Returns:
        NDCG@K
    """
    ndcg = 0.0
    for pred, gt in zip(predictions, ground_truth):
        if gt in pred[:k]:
            rank = pred[:k].index(gt)
            ndcg += 1.0 / np.log2(rank + 2)
    return ndcg / len(ground_truth) if ground_truth else 0.0


def recall_at_k(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k: int = 10
) -> float:
    """
    计算 Recall@K
    
    Args:
        predictions: 预测列表
        ground_truth: 真实物品列表（每个用户可能有多个）
        k: 截断位置
    
    Returns:
        Recall@K
    """
    recalls = []
    for pred, gt_list in zip(predictions, ground_truth):
        if not gt_list:
            continue
        hits = len(set(pred[:k]) & set(gt_list))
        recalls.append(hits / len(gt_list))
    return np.mean(recalls) if recalls else 0.0


def precision_at_k(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k: int = 10
) -> float:
    """
    计算 Precision@K
    
    Args:
        predictions: 预测列表
        ground_truth: 真实物品列表
        k: 截断位置
    
    Returns:
        Precision@K
    """
    precisions = []
    for pred, gt_list in zip(predictions, ground_truth):
        if not pred[:k]:
            continue
        hits = len(set(pred[:k]) & set(gt_list))
        precisions.append(hits / k)
    return np.mean(precisions) if precisions else 0.0


def accuracy(
    predictions: List[int],
    ground_truth: List[int]
) -> float:
    """
    计算准确率
    
    Args:
        predictions: 预测物品列表
        ground_truth: 真实物品列表
    
    Returns:
        准确率
    """
    if not ground_truth:
        return 0.0
    
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(ground_truth)


class RecommendationEvaluator:
    """
    推荐系统评估器
    """
    
    def __init__(self, k_list: List[int] = [3, 5, 10]):
        self.k_list = k_list
        self.metrics = {}
    
    def evaluate(
        self,
        predictions: List[List[int]],
        ground_truth: List[int],
        ground_truth_list: List[List[int]] = None
    ) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            predictions: 预测列表
            ground_truth: 单物品真实标签
            ground_truth_list: 多物品真实标签
        
        Returns:
            指标字典
        """
        results = {}
        
        for k in self.k_list:
            results[f"Hit@{k}"] = hit_rate(predictions, ground_truth, k)
            results[f"NDCG@{k}"] = ndcg_at_k(predictions, ground_truth, k)
            
            if ground_truth_list:
                results[f"Recall@{k}"] = recall_at_k(predictions, ground_truth_list, k)
                results[f"Precision@{k}"] = precision_at_k(predictions, ground_truth_list, k)
        
        return results
    
    def print_results(self, results: Dict[str, float]):
        """打印评估结果"""
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        for metric, value in sorted(results.items()):
            print(f"{metric}: {value:.4f}")
        print("=" * 50)
