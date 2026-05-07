"""
超参数调优脚本
系统性地测试不同超参数组合

对应论文 Section 4.5 (Parameter Analysis)
"""

import os
import argparse
import json
from typing import Dict, List, Tuple
import itertools

import torch
import matplotlib.pyplot as plt
import numpy as np

from configs import Config, get_config
from main import K_RagRec


class HyperparameterTuner:
    """超参数调优器"""
    
    def __init__(self, config: Config, device: str = "cuda"):
        self.config = config
        self.device = device
        self.results = {}
    
    def tune_popularity_threshold(
        self,
        kg_dir: str,
        test_data: List,
        thresholds: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
    ) -> Dict:
        """
        调优流行度阈值 p
        对应论文 Figure 4
        """
        print("\n" + "="*60)
        print("调优流行度阈值 p (对应论文 Figure 4)")
        print("="*60)
        
        results = {
            "thresholds": [],
            "accuracy": [],
            "recall@5": [],
            "inference_time": []
        }
        
        for p in thresholds:
            print(f"\n测试 p = {p}")
            
            # 更新配置
            self.config.retrieval.popularity_threshold = p
            
            # 运行实验
            model = K_RagRec(self.config, device=self.device)
            model.initialize(kg_dir, use_sample_data=False)
            
            # 评估（简化版）
            metrics = self._evaluate(model, test_data)
            
            results["thresholds"].append(p)
            results["accuracy"].append(metrics["accuracy"])
            results["recall@5"].append(metrics["recall@5"])
            results["inference_time"].append(metrics["inference_time"])
            
            print(f"  ACC: {metrics['accuracy']:.4f}")
            print(f"  R@5: {metrics['recall@5']:.4f}")
            print(f"  时间: {metrics['inference_time']:.4f}s")
        
        self.results["popularity_threshold"] = results
        return results
    
    def tune_retrieval_numbers(
        self,
        kg_dir: str,
        test_data: List,
        k_values: List[int] = [1, 3, 5, 7, 10],
        n_values: List[int] = [3, 5, 7, 10]
    ) -> Dict:
        """
        调优检索数量 K 和重排序数量 N
        对应论文 Figure 5 和 Figure 6
        """
        print("\n" + "="*60)
        print("调优检索数量 K 和重排序数量 N")
        print("="*60)
        
        # 调优 K (固定 N=5)
        print("\n[1/2] 调优 K (固定 N=5)")
        k_results = {
            "k_values": [],
            "accuracy": [],
            "recall@3": [],
            "recall@5": []
        }
        
        for k in k_values:
            print(f"\n测试 K = {k}")
            
            self.config.retrieval.topk_nodes = k
            self.config.retrieval.topk_rerank = 5
            
            model = K_RagRec(self.config, device=self.device)
            model.initialize(kg_dir, use_sample_data=False)
            
            metrics = self._evaluate(model, test_data)
            
            k_results["k_values"].append(k)
            k_results["accuracy"].append(metrics["accuracy"])
            k_results["recall@3"].append(metrics["recall@3"])
            k_results["recall@5"].append(metrics["recall@5"])
            
            print(f"  ACC: {metrics['accuracy']:.4f}")
        
        # 调优 N (固定 K=3)
        print("\n[2/2] 调优 N (固定 K=3)")
        n_results = {
            "n_values": [],
            "accuracy": [],
            "recall@3": [],
            "recall@5": []
        }
        
        for n in n_values:
            print(f"\n测试 N = {n}")
            
            self.config.retrieval.topk_nodes = 3
            self.config.retrieval.topk_rerank = n
            
            model = K_RagRec(self.config, device=self.device)
            model.initialize(kg_dir, use_sample_data=False)
            
            metrics = self._evaluate(model, test_data)
            
            n_results["n_values"].append(n)
            n_results["accuracy"].append(metrics["accuracy"])
            n_results["recall@3"].append(metrics["recall@3"])
            n_results["recall@5"].append(metrics["recall@5"])
            
            print(f"  ACC: {metrics['accuracy']:.4f}")
        
        self.results["topk_retrieval"] = k_results
        self.results["topk_rerank"] = n_results
        
        return {"k_results": k_results, "n_results": n_results}
    
    def tune_gnn_layers(
        self,
        kg_dir: str,
        test_data: List,
        layer_values: List[int] = [2, 3, 4, 5]
    ) -> Dict:
        """
        调优 GNN 层数
        对应论文 Table 10
        """
        print("\n" + "="*60)
        print("调优 GNN 层数 (对应论文 Table 10)")
        print("="*60)
        
        results = {
            "layers": [],
            "accuracy": [],
            "recall@3": [],
            "recall@5": []
        }
        
        for layers in layer_values:
            print(f"\n测试 GNN 层数 = {layers}")
            
            self.config.model.gnn_num_layers = layers
            
            model = K_RagRec(self.config, device=self.device)
            model.initialize(kg_dir, use_sample_data=False)
            
            metrics = self._evaluate(model, test_data)
            
            results["layers"].append(layers)
            results["accuracy"].append(metrics["accuracy"])
            results["recall@3"].append(metrics["recall@3"])
            results["recall@5"].append(metrics["recall@5"])
            
            print(f"  ACC: {metrics['accuracy']:.4f}")
            print(f"  R@3: {metrics['recall@3']:.4f}")
            print(f"  R@5: {metrics['recall@5']:.4f}")
        
        self.results["gnn_layers"] = results
        return results
    
    def _evaluate(self, model: K_RagRec, test_data: List) -> Dict:
        """评估模型（简化版）"""
        # 这里需要实现实际的评估逻辑
        # 返回模拟结果
        return {
            "accuracy": 0.435,
            "recall@3": 0.725,
            "recall@5": 0.831,
            "inference_time": 0.5
        }
    
    def plot_results(self, output_dir: str):
        """绘制结果图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 绘制流行度阈值影响
        if "popularity_threshold" in self.results:
            self._plot_popularity_threshold(output_dir)
        
        # 绘制检索数量影响
        if "topk_retrieval" in self.results:
            self._plot_topk_retrieval(output_dir)
        
        if "topk_rerank" in self.results:
            self._plot_topk_rerank(output_dir)
        
        # 绘制 GNN 层数影响
        if "gnn_layers" in self.results:
            self._plot_gnn_layers(output_dir)
        
        print(f"\n图表已保存到: {output_dir}")
    
    def _plot_popularity_threshold(self, output_dir: str):
        """绘制流行度阈值影响（对应论文 Figure 4）"""
        data = self.results["popularity_threshold"]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # ACC
        axes[0].plot(data["thresholds"], data["accuracy"], 'o-', linewidth=2)
        axes[0].set_xlabel("Popularity Threshold p")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("(a) ML1M ACC")
        axes[0].grid(True, alpha=0.3)
        
        # Recall@5
        axes[1].plot(data["thresholds"], data["recall@5"], 'o-', linewidth=2)
        axes[1].set_xlabel("Popularity Threshold p")
        axes[1].set_ylabel("Recall@5")
        axes[1].set_title("(b) ML1M R@5")
        axes[1].grid(True, alpha=0.3)
        
        # Inference Time
        axes[2].plot(data["thresholds"], data["inference_time"], 'o-', linewidth=2)
        axes[2].set_xlabel("Popularity Threshold p")
        axes[2].set_ylabel("Inference Time (s)")
        axes[2].set_title("(c) ML1M Inference Time")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "popularity_threshold.png"), dpi=300)
        plt.close()
    
    def _plot_topk_retrieval(self, output_dir: str):
        """绘制检索数量 K 影响（对应论文 Figure 5）"""
        data = self.results["topk_retrieval"]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(data["k_values"], data["accuracy"], 'o-', linewidth=2)
        axes[0].set_xlabel("Top-K Retrieval")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("(a) MovieLens ACC")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(data["k_values"], data["recall@3"], 'o-', linewidth=2)
        axes[1].set_xlabel("Top-K Retrieval")
        axes[1].set_ylabel("Recall@3")
        axes[1].set_title("(b) MovieLens R@3")
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(data["k_values"], data["recall@5"], 'o-', linewidth=2)
        axes[2].set_xlabel("Top-K Retrieval")
        axes[2].set_ylabel("Recall@5")
        axes[2].set_title("(c) MovieLens R@5")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "topk_retrieval.png"), dpi=300)
        plt.close()
    
    def _plot_topk_rerank(self, output_dir: str):
        """绘制重排序数量 N 影响（对应论文 Figure 6）"""
        data = self.results["topk_rerank"]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(data["n_values"], data["accuracy"], 'o-', linewidth=2)
        axes[0].set_xlabel("Top-N Re-ranking")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("(a) Amazon Book ACC")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(data["n_values"], data["recall@3"], 'o-', linewidth=2)
        axes[1].set_xlabel("Top-N Re-ranking")
        axes[1].set_ylabel("Recall@3")
        axes[1].set_title("(b) Amazon Book R@3")
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(data["n_values"], data["recall@5"], 'o-', linewidth=2)
        axes[2].set_xlabel("Top-N Re-ranking")
        axes[2].set_ylabel("Recall@5")
        axes[2].set_title("(c) Amazon Book R@5")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "topk_rerank.png"), dpi=300)
        plt.close()
    
    def _plot_gnn_layers(self, output_dir: str):
        """绘制 GNN 层数影响（对应论文 Table 10）"""
        data = self.results["gnn_layers"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(data["layers"]))
        width = 0.25
        
        ax.bar(x - width, data["accuracy"], width, label='Accuracy')
        ax.bar(x, data["recall@3"], width, label='Recall@3')
        ax.bar(x + width, data["recall@5"], width, label='Recall@5')
        
        ax.set_xlabel('GNN Layers')
        ax.set_ylabel('Performance')
        ax.set_title('Effect of GNN Layer Numbers')
        ax.set_xticks(x)
        ax.set_xticklabels(data["layers"])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gnn_layers.png"), dpi=300)
        plt.close()
    
    def save_results(self, output_dir: str):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "hyperparameter_tuning_results.json")
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="K-RagRec 超参数调优")
    
    parser.add_argument("--data_dir", type=str, default="data/ml-1m")
    parser.add_argument("--kg_dir", type=str, default="data/kg")
    parser.add_argument("--output_dir", type=str, default="results/tuning")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tune_all", action="store_true",
                        help="调优所有超参数")
    parser.add_argument("--tune_popularity", action="store_true",
                        help="调优流行度阈值")
    parser.add_argument("--tune_retrieval", action="store_true",
                        help="调优检索数量")
    parser.add_argument("--tune_gnn", action="store_true",
                        help="调优 GNN 层数")
    
    args = parser.parse_args()
    
    print("="*60)
    print("K-RagRec 超参数调优")
    print("对应论文 Section 4.5")
    print("="*60)
    
    # 加载配置
    config = get_config()
    
    # 加载测试数据
    test_data = []  # 需要实现实际的数据加载
    
    # 创建调优器
    tuner = HyperparameterTuner(config, device=args.device)
    
    kg_dir = os.path.join(os.path.dirname(__file__), args.kg_dir)
    
    # 运行调优
    if args.tune_all or args.tune_popularity:
        tuner.tune_popularity_threshold(kg_dir, test_data)
    
    if args.tune_all or args.tune_retrieval:
        tuner.tune_retrieval_numbers(kg_dir, test_data)
    
    if args.tune_all or args.tune_gnn:
        tuner.tune_gnn_layers(kg_dir, test_data)
    
    # 绘制结果
    tuner.plot_results(args.output_dir)
    
    # 保存结果
    tuner.save_results(args.output_dir)
    
    print("\n" + "="*60)
    print("超参数调优完成!")
    print("="*60)
    print(f"\n查看结果: {args.output_dir}")


if __name__ == "__main__":
    main()
