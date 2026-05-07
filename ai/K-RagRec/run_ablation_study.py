"""
Ablation Study 脚本
系统性地测试各个模块的贡献

对应论文 Section 4.3 和 Figure 3
"""

import os
import argparse
import json
from datetime import datetime
from typing import Dict, List

import torch
from tqdm import tqdm

from configs import Config, get_config
from main import K_RagRec
from utils import RecommendationEvaluator


class AblationStudy:
    """Ablation Study 实验管理器"""
    
    def __init__(self, config: Config, device: str = "cuda"):
        self.config = config
        self.device = device
        self.results = {}
    
    def run_variant(
        self,
        variant_name: str,
        kg_dir: str,
        test_data: List,
        **kwargs
    ) -> Dict:
        """运行一个 ablation 变体"""
        print("\n" + "="*60)
        print(f"运行变体: {variant_name}")
        print("="*60)
        
        # 创建模型
        model = K_RagRec(self.config, device=self.device)
        model.initialize(kg_dir, use_sample_data=False)
        
        # 应用变体配置
        if "no_gnn_indexing" in kwargs and kwargs["no_gnn_indexing"]:
            print("  [变体] 禁用 GNN Indexing")
            model.indexer.layer_embeddings = {}
        
        if "no_adaptive" in kwargs and kwargs["no_adaptive"]:
            print("  [变体] 禁用自适应检索")
            model.adaptive_retriever.popularity_threshold = 1.0
        
        if "no_reranking" in kwargs and kwargs["no_reranking"]:
            print("  [变体] 禁用重排序")
            self.config.retrieval.topk_rerank = self.config.retrieval.topk_nodes
        
        if "no_gnn_encoding" in kwargs and kwargs["no_gnn_encoding"]:
            print("  [变体] 禁用 GNN Encoding")
            model.gnn_encoder = None
        
        # 评估
        evaluator = RecommendationEvaluator()
        
        print("\n开始评估...")
        for user_history, target_item in tqdm(test_data[:100]):  # 使用前100个样本快速测试
            try:
                recommendation = model.recommend(
                    user_history=user_history,
                    num_recommendations=5
                )
                # 这里需要解析推荐结果并计算指标
                # 简化版本，实际需要更复杂的解析逻辑
            except Exception as e:
                print(f"  警告: 推荐失败 - {e}")
                continue
        
        # 计算指标
        metrics = {
            "accuracy": 0.0,  # 需要实际计算
            "recall@3": 0.0,
            "recall@5": 0.0,
            "inference_time": 0.0
        }
        
        print(f"\n结果:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def run_all_variants(self, kg_dir: str, test_data: List):
        """运行所有 ablation 变体"""
        print("\n" + "="*60)
        print("Ablation Study - 测试所有变体")
        print("="*60)
        
        variants = {
            "K-RagRec (完整)": {},
            "K-RagRec (-Indexing)": {"no_gnn_indexing": True},
            "K-RagRec (-Adaptive)": {"no_adaptive": True},
            "K-RagRec (-Reranking)": {"no_reranking": True},
            "K-RagRec (-Encoding)": {"no_gnn_encoding": True},
        }
        
        for variant_name, variant_config in variants.items():
            self.results[variant_name] = self.run_variant(
                variant_name,
                kg_dir,
                test_data,
                **variant_config
            )
        
        return self.results
    
    def save_results(self, output_dir: str):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"ablation_results_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n结果已保存到: {output_file}")
    
    def print_comparison_table(self):
        """打印对比表格"""
        print("\n" + "="*60)
        print("Ablation Study 结果对比")
        print("="*60)
        
        # 表头
        print(f"\n{'变体':<30} {'ACC':<10} {'R@3':<10} {'R@5':<10}")
        print("-" * 60)
        
        # 数据行
        for variant_name, metrics in self.results.items():
            print(f"{variant_name:<30} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['recall@3']:<10.4f} "
                  f"{metrics['recall@5']:<10.4f}")
        
        # 计算性能下降
        if "K-RagRec (完整)" in self.results:
            baseline = self.results["K-RagRec (完整)"]
            print("\n" + "="*60)
            print("相对完整模型的性能下降")
            print("="*60)
            print(f"\n{'变体':<30} {'ACC ↓':<10} {'R@3 ↓':<10} {'R@5 ↓':<10}")
            print("-" * 60)
            
            for variant_name, metrics in self.results.items():
                if variant_name == "K-RagRec (完整)":
                    continue
                
                acc_drop = (baseline['accuracy'] - metrics['accuracy']) / baseline['accuracy'] * 100
                r3_drop = (baseline['recall@3'] - metrics['recall@3']) / baseline['recall@3'] * 100
                r5_drop = (baseline['recall@5'] - metrics['recall@5']) / baseline['recall@5'] * 100
                
                print(f"{variant_name:<30} "
                      f"{acc_drop:<10.1f}% "
                      f"{r3_drop:<10.1f}% "
                      f"{r5_drop:<10.1f}%")


def load_test_data(data_dir: str) -> List:
    """加载测试数据"""
    # 这里需要实现实际的数据加载逻辑
    # 返回格式: [(user_history, target_item), ...]
    
    print(f"从 {data_dir} 加载测试数据...")
    
    # 示例数据
    test_data = [
        (["Toy Story", "Jumanji", "The Matrix"], "Inception"),
        (["The Matrix", "Inception", "Interstellar"], "The Dark Knight"),
        # ... 更多测试样本
    ]
    
    print(f"加载了 {len(test_data)} 个测试样本")
    return test_data


def main():
    parser = argparse.ArgumentParser(description="K-RagRec Ablation Study")
    
    parser.add_argument("--data_dir", type=str, default="data/ml-1m",
                        help="数据目录")
    parser.add_argument("--kg_dir", type=str, default="data/kg",
                        help="知识图谱目录")
    parser.add_argument("--output_dir", type=str, default="results/ablation",
                        help="结果输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="测试样本数量")
    
    args = parser.parse_args()
    
    print("="*60)
    print("K-RagRec Ablation Study")
    print("对应论文 Section 4.3")
    print("="*60)
    
    # 加载配置
    config = get_config()
    
    # 加载测试数据
    test_data = load_test_data(args.data_dir)
    
    # 创建 Ablation Study 管理器
    study = AblationStudy(config, device=args.device)
    
    # 运行所有变体
    kg_dir = os.path.join(os.path.dirname(__file__), args.kg_dir)
    study.run_all_variants(kg_dir, test_data[:args.num_samples])
    
    # 打印对比表格
    study.print_comparison_table()
    
    # 保存结果
    study.save_results(args.output_dir)
    
    print("\n" + "="*60)
    print("Ablation Study 完成!")
    print("="*60)
    
    print("\n论文对比 (MovieLens-1M, LLama-2-7B):")
    print("  完整模型 ACC: 0.435")
    print("  去掉 GNN Encoding: ACC 下降 ~37%")
    print("\n你的结果见上表 ↑")


if __name__ == "__main__":
    main()
