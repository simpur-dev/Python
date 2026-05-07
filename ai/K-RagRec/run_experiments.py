"""
完整实验运行脚本
自动运行训练、评估和对比实验

对应论文 Table 1 的完整实验流程
"""

import os
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List

import torch
from tqdm import tqdm

from configs import Config, get_config
from main import K_RagRec


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config: Config, device: str = "cuda"):
        self.config = config
        self.device = device
        self.results = {}
    
    def run_baseline(self, name: str, kg_dir: str, test_data: List) -> Dict:
        """运行基线方法"""
        print(f"\n运行基线: {name}")
        
        # 这里需要实现各个基线方法
        # 简化版本，返回模拟结果
        metrics = {
            "accuracy": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "inference_time": 0.0
        }
        
        return metrics
    
    def run_kragrec(
        self,
        kg_dir: str,
        test_data: List,
        mode: str = "prompt_tuning"
    ) -> Dict:
        """运行 K-RagRec"""
        print(f"\n运行 K-RagRec ({mode})")
        
        model = K_RagRec(self.config, device=self.device)
        model.initialize(kg_dir, use_sample_data=False)
        
        # 评估
        start_time = time.time()
        
        correct = 0
        recall_3 = 0
        recall_5 = 0
        total = len(test_data)
        
        print("开始评估...")
        for user_history, target_item in tqdm(test_data):
            try:
                recommendation = model.recommend(
                    user_history=user_history,
                    num_recommendations=5
                )
                
                # 简化的评估逻辑
                # 实际需要解析 LLM 输出并匹配目标物品
                
            except Exception as e:
                print(f"警告: {e}")
                continue
        
        inference_time = (time.time() - start_time) / total
        
        metrics = {
            "accuracy": correct / total if total > 0 else 0.0,
            "recall@3": recall_3 / total if total > 0 else 0.0,
            "recall@5": recall_5 / total if total > 0 else 0.0,
            "inference_time": inference_time
        }
        
        print(f"\n结果:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def run_full_comparison(self, kg_dir: str, test_data: List):
        """运行完整对比实验（对应论文 Table 1）"""
        print("\n" + "="*60)
        print("完整对比实验 (对应论文 Table 1)")
        print("="*60)
        
        experiments = {
            # Inference-only 方法
            "KG-Text": {"type": "baseline", "name": "KG-Text"},
            "KAPING": {"type": "baseline", "name": "KAPING"},
            
            # Prompt Tuning 方法
            "PT w/ KG-Text": {"type": "baseline", "name": "PT w/ KG-Text"},
            "GraphToken w/ RAG": {"type": "baseline", "name": "GraphToken w/ RAG"},
            "G-retriever": {"type": "baseline", "name": "G-retriever"},
            "K-RagRec": {"type": "kragrec", "mode": "prompt_tuning"},
            
            # Fine-tuning 方法
            "Lora w/ KG-Text": {"type": "baseline", "name": "Lora w/ KG-Text"},
            "Lora w/ K-RagRec": {"type": "kragrec", "mode": "lora"},
        }
        
        for exp_name, exp_config in experiments.items():
            if exp_config["type"] == "baseline":
                self.results[exp_name] = self.run_baseline(
                    exp_config["name"],
                    kg_dir,
                    test_data
                )
            elif exp_config["type"] == "kragrec":
                self.results[exp_name] = self.run_kragrec(
                    kg_dir,
                    test_data,
                    mode=exp_config["mode"]
                )
        
        return self.results
    
    def print_comparison_table(self):
        """打印对比表格（论文 Table 1 格式）"""
        print("\n" + "="*60)
        print("实验结果对比 (对应论文 Table 1)")
        print("="*60)
        
        print(f"\n{'方法':<25} {'ACC':<10} {'R@3':<10} {'R@5':<10} {'时间(s)':<10}")
        print("-" * 65)
        
        # Inference-only
        print("\nInference-only:")
        for name in ["KG-Text", "KAPING"]:
            if name in self.results:
                m = self.results[name]
                print(f"{name:<25} {m['accuracy']:<10.3f} {m['recall@3']:<10.3f} "
                      f"{m['recall@5']:<10.3f} {m['inference_time']:<10.3f}")
        
        # Prompt Tuning
        print("\nFrozen LLM w/ Prompt Tuning:")
        for name in ["PT w/ KG-Text", "GraphToken w/ RAG", "G-retriever", "K-RagRec"]:
            if name in self.results:
                m = self.results[name]
                print(f"{name:<25} {m['accuracy']:<10.3f} {m['recall@3']:<10.3f} "
                      f"{m['recall@5']:<10.3f} {m['inference_time']:<10.3f}")
        
        # Fine-tuning
        print("\nFine-tuning:")
        for name in ["Lora w/ KG-Text", "Lora w/ K-RagRec"]:
            if name in self.results:
                m = self.results[name]
                print(f"{name:<25} {m['accuracy']:<10.3f} {m['recall@3']:<10.3f} "
                      f"{m['recall@5']:<10.3f} {m['inference_time']:<10.3f}")
        
        # 计算改进
        if "K-RagRec" in self.results and "G-retriever" in self.results:
            kragrec = self.results["K-RagRec"]
            baseline = self.results["G-retriever"]
            
            acc_imp = (kragrec['accuracy'] - baseline['accuracy']) / baseline['accuracy'] * 100
            r3_imp = (kragrec['recall@3'] - baseline['recall@3']) / baseline['recall@3'] * 100
            r5_imp = (kragrec['recall@5'] - baseline['recall@5']) / baseline['recall@5'] * 100
            
            print(f"\n{'K-RagRec 改进':<25} {acc_imp:<10.1f}% {r3_imp:<10.1f}% {r5_imp:<10.1f}%")
        
        # 论文报告值
        print("\n" + "="*60)
        print("论文报告值 (MovieLens-1M, LLama-2-7B)")
        print("="*60)
        print(f"{'K-RagRec':<25} {'0.435':<10} {'0.725':<10} {'0.831':<10}")
        print(f"{'G-retriever':<25} {'0.274':<10} {'0.532':<10} {'0.650':<10}")
        print(f"{'改进':<25} {'58.6%':<10} {'33.0%':<10} {'27.8%':<10}")
    
    def save_results(self, output_dir: str):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"experiment_results_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="K-RagRec 完整实验")
    
    parser.add_argument("--data_dir", type=str, default="data/ml-1m",
                        help="数据目录")
    parser.add_argument("--kg_dir", type=str, default="data/kg",
                        help="知识图谱目录")
    parser.add_argument("--output_dir", type=str, default="results/experiments",
                        help="结果输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="LLM 模型")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="测试样本数量")
    
    args = parser.parse_args()
    
    print("="*60)
    print("K-RagRec 完整实验")
    print("对应论文 Table 1")
    print("="*60)
    print(f"\n配置:")
    print(f"  数据集: {args.data_dir}")
    print(f"  LLM: {args.llm_model}")
    print(f"  设备: {args.device}")
    print(f"  测试样本: {args.num_samples}")
    
    # 加载配置
    config = get_config()
    config.model.llm_model_name = args.llm_model
    
    # 加载测试数据
    print(f"\n加载测试数据...")
    # 这里需要实现实际的数据加载
    test_data = []  # [(user_history, target_item), ...]
    
    # 创建实验运行器
    runner = ExperimentRunner(config, device=args.device)
    
    # 运行完整对比实验
    kg_dir = os.path.join(os.path.dirname(__file__), args.kg_dir)
    runner.run_full_comparison(kg_dir, test_data[:args.num_samples])
    
    # 打印对比表格
    runner.print_comparison_table()
    
    # 保存结果
    runner.save_results(args.output_dir)
    
    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)


if __name__ == "__main__":
    main()
