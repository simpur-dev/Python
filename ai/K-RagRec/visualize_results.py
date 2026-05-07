"""
结果可视化脚本
生成论文风格的图表
"""

import os
import json
import argparse
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300


class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.results = {}
    
    def load_results(self, filename: str):
        """加载结果文件"""
        filepath = os.path.join(self.results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    
    def plot_main_results(self, output_dir: str):
        """
        绘制主要结果对比图（对应论文 Table 1）
        """
        print("\n绘制主要结果对比图...")
        
        # 论文报告的结果
        methods = [
            "KG-Text",
            "KAPING",
            "PT w/ KG-Text",
            "GraphToken\nw/ RAG",
            "G-retriever",
            "K-RagRec"
        ]
        
        # MovieLens-1M, LLama-2-7B
        acc = [0.076, 0.079, 0.078, 0.268, 0.274, 0.435]
        r3 = [0.0, 0.0, 0.191, 0.421, 0.532, 0.725]
        r5 = [0.0, 0.0, 0.308, 0.466, 0.650, 0.831]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        x = np.arange(len(methods))
        width = 0.6
        
        # Accuracy
        bars1 = axes[0].bar(x, acc, width, color='steelblue', alpha=0.8)
        axes[0].set_ylabel('Accuracy', fontsize=14)
        axes[0].set_title('(a) Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0].set_ylim([0, 0.5])
        
        # 标注最高值
        axes[0].bar(5, acc[5], width, color='coral', alpha=0.8)
        axes[0].text(5, acc[5] + 0.02, f'{acc[5]:.3f}', ha='center', fontweight='bold')
        
        # Recall@3
        bars2 = axes[1].bar(x[2:], r3[2:], width, color='steelblue', alpha=0.8)
        axes[1].set_ylabel('Recall@3', fontsize=14)
        axes[1].set_title('(b) Recall@3 Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x[2:])
        axes[1].set_xticklabels(methods[2:], rotation=45, ha='right')
        axes[1].set_ylim([0, 0.8])
        
        axes[1].bar(3, r3[5], width, color='coral', alpha=0.8)
        axes[1].text(3, r3[5] + 0.03, f'{r3[5]:.3f}', ha='center', fontweight='bold')
        
        # Recall@5
        bars3 = axes[2].bar(x[2:], r5[2:], width, color='steelblue', alpha=0.8)
        axes[2].set_ylabel('Recall@5', fontsize=14)
        axes[2].set_title('(c) Recall@5 Comparison', fontsize=14, fontweight='bold')
        axes[2].set_xticks(x[2:])
        axes[2].set_xticklabels(methods[2:], rotation=45, ha='right')
        axes[2].set_ylim([0, 0.9])
        
        axes[2].bar(3, r5[5], width, color='coral', alpha=0.8)
        axes[2].text(3, r5[5] + 0.03, f'{r5[5]:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "main_results_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  保存到: {output_dir}/main_results_comparison.png")
    
    def plot_ablation_results(self, output_dir: str):
        """
        绘制 Ablation Study 结果（对应论文 Figure 3）
        """
        print("\n绘制 Ablation Study 结果...")
        
        variants = [
            "K-RagRec\n(完整)",
            "K-RagRec\n(-Indexing)",
            "K-RagRec\n(-Popularity)",
            "K-RagRec\n(-Reranking)",
            "K-RagRec\n(-Encoding)"
        ]
        
        # MovieLens-1M 数据（论文 Figure 3）
        ml_acc = [0.435, 0.274, 0.420, 0.410, 0.274]
        ml_r3 = [0.725, 0.532, 0.710, 0.695, 0.532]
        ml_r5 = [0.831, 0.650, 0.820, 0.810, 0.650]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        x = np.arange(len(variants))
        width = 0.6
        
        colors = ['coral', 'steelblue', 'steelblue', 'steelblue', 'steelblue']
        
        # ML-1M ACC
        axes[0].bar(x, ml_acc, width, color=colors, alpha=0.8)
        axes[0].set_ylabel('Accuracy', fontsize=14)
        axes[0].set_title('(a) ML1M ACC', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(variants, rotation=45, ha='right')
        axes[0].set_ylim([0, 0.5])
        
        # ML-1M R@3
        axes[1].bar(x, ml_r3, width, color=colors, alpha=0.8)
        axes[1].set_ylabel('Recall@3', fontsize=14)
        axes[1].set_title('(b) ML1M R@3', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(variants, rotation=45, ha='right')
        axes[1].set_ylim([0, 0.8])
        
        # ML-1M R@5
        axes[2].bar(x, ml_r5, width, color=colors, alpha=0.8)
        axes[2].set_ylabel('Recall@5', fontsize=14)
        axes[2].set_title('(c) ML1M R@5', fontsize=14, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(variants, rotation=45, ha='right')
        axes[2].set_ylim([0, 0.9])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ablation_study.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  保存到: {output_dir}/ablation_study.png")
    
    def plot_efficiency_comparison(self, output_dir: str):
        """
        绘制效率对比图（对应论文 Table 2）
        """
        print("\n绘制效率对比图...")
        
        methods = [
            "Direct\nInference",
            "KG-Text",
            "KAPING",
            "GraphToken\nw/ RAG",
            "G-retriever",
            "K-RagRec"
        ]
        
        # 推理时间（秒）
        times = [0.42, 1.85, 1.92, 1.78, 1.81, 0.52]
        
        # Accuracy
        acc = [0.065, 0.076, 0.079, 0.268, 0.274, 0.435]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        x = np.arange(len(methods))
        width = 0.6
        
        # 推理时间
        colors = ['lightgray', 'steelblue', 'steelblue', 'steelblue', 'steelblue', 'coral']
        axes[0].bar(x, times, width, color=colors, alpha=0.8)
        axes[0].set_ylabel('Inference Time (s)', fontsize=14)
        axes[0].set_title('(a) Inference Time Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0].axhline(y=0.52, color='red', linestyle='--', alpha=0.5, label='K-RagRec')
        axes[0].legend()
        
        # 效率-性能权衡
        axes[1].scatter(times[:-1], acc[:-1], s=200, alpha=0.6, color='steelblue', label='Baselines')
        axes[1].scatter(times[-1], acc[-1], s=300, alpha=0.8, color='coral', marker='*', label='K-RagRec')
        
        for i, method in enumerate(methods):
            axes[1].annotate(method.replace('\n', ' '), (times[i], acc[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[1].set_xlabel('Inference Time (s)', fontsize=14)
        axes[1].set_ylabel('Accuracy', fontsize=14)
        axes[1].set_title('(b) Efficiency-Performance Trade-off', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "efficiency_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  保存到: {output_dir}/efficiency_comparison.png")
    
    def plot_dataset_comparison(self, output_dir: str):
        """
        绘制不同数据集上的性能对比
        """
        print("\n绘制数据集对比图...")
        
        datasets = ['ML-1M', 'ML-20M', 'Amazon\nBook']
        
        # K-RagRec 在不同数据集上的表现
        acc = [0.435, 0.600, 0.508]
        r3 = [0.725, 0.850, 0.690]
        r5 = [0.831, 0.913, 0.780]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(datasets))
        width = 0.25
        
        bars1 = ax.bar(x - width, acc, width, label='Accuracy', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x, r3, width, label='Recall@3', color='coral', alpha=0.8)
        bars3 = ax.bar(x + width, r5, width, label='Recall@5', color='lightgreen', alpha=0.8)
        
        ax.set_ylabel('Performance', fontsize=14)
        ax.set_title('K-RagRec Performance Across Datasets', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend(fontsize=12)
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dataset_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  保存到: {output_dir}/dataset_comparison.png")
    
    def generate_all_plots(self, output_dir: str):
        """生成所有图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*60)
        print("生成论文风格图表")
        print("="*60)
        
        self.plot_main_results(output_dir)
        self.plot_ablation_results(output_dir)
        self.plot_efficiency_comparison(output_dir)
        self.plot_dataset_comparison(output_dir)
        
        print("\n" + "="*60)
        print("所有图表生成完成!")
        print("="*60)
        print(f"\n查看结果: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="K-RagRec 结果可视化")
    
    parser.add_argument("--results_dir", type=str, default="results",
                        help="结果目录")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="图表输出目录")
    
    args = parser.parse_args()
    
    visualizer = ResultVisualizer(args.results_dir)
    visualizer.generate_all_plots(args.output_dir)


if __name__ == "__main__":
    main()
