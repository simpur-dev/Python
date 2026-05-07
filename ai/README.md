# K-RagRec: Knowledge Graph Retrieval-Augmented Recommendation

[![Paper](https://img.shields.io/badge/Paper-ACL%202025-blue)](https://arxiv.org/abs/2501.02226)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

基于知识图谱检索增强的大语言模型推荐系统框架

**论文**: [Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation](https://arxiv.org/abs/2501.02226)  
**会议**: ACL 2025 Main Conference  
**作者**: Shijie Wang, Wenqi Fan, et al.

---

## 🌟 核心特性

- ✅ **分层知识子图索引**: 使用 GNN 构建多层次知识表示
- ✅ **自适应检索策略**: 基于流行度动态决定是否检索，提升效率 50%
- ✅ **知识子图重排序**: 精细化排序确保最相关知识优先
- ✅ **结构化知识增强**: 避免文本 RAG 的噪音问题
- ✅ **即插即用**: 支持 LLama-2/3, QWEN2 等多种 LLM

## 📊 性能表现

在 MovieLens-1M 数据集上（LLama-2-7B）：

| 方法 | Accuracy | Recall@3 | Recall@5 |
|------|----------|----------|----------|
| G-retriever | 0.274 | 0.532 | 0.650 |
| **K-RagRec** | **0.435** | **0.725** | **0.831** |
| **提升** | **+58.6%** | **+33.0%** | **+27.8%** |

---

## 🚀 快速开始

### 方法 1: 一键启动（推荐）

**Linux/Mac:**
```bash
chmod +x K-RagRec/quick_start.sh
./K-RagRec/quick_start.sh
```

**Windows:**
```cmd
K-RagRec\quick_start.bat
```

### 方法 2: 手动步骤

#### 1. 环境检查
```bash
python check_env.py
```

#### 2. 安装依赖
```bash
cd K-RagRec
pip install -r requirements.txt
```

#### 3. 下载数据
```bash
python download_data.py
```

#### 4. 运行测试
```bash
python run_full_test.py
```

#### 5. 运行 Demo
```bash
python main.py --mode demo --use_sample_data
```

---

## 📁 项目结构

```
K-RagRec/
├── configs/              # 配置文件
│   └── config.py        # 模型和训练配置
├── model/               # 模型实现
│   ├── gnn_encoder.py   # GNN 编码器
│   └── projector.py     # 投影层
├── indexing/            # 知识图谱索引
│   └── index_KG.py      # 分层索引构建
├── retrieval/           # 检索模块
│   ├── retriever.py     # 自适应检索器
│   └── retrieve.py      # 子图检索和重排序
├── generation/          # 生成模块
│   └── generator.py     # LLM 推荐生成
├── utils/               # 工具函数
│   ├── data_processor.py # 数据处理
│   └── metrics.py       # 评估指标
├── data/                # 数据目录
│   ├── ml-1m/          # MovieLens-1M
│   └── kg/             # 知识图谱
├── main.py              # 主入口
├── run_full_test.py     # 完整测试
├── run_ablation_study.py # Ablation Study
├── run_experiments.py   # 完整实验
├── tune_hyperparameters.py # 超参数调优
└── download_data.py     # 数据下载
```

---

## 🔬 实验复现

### 1. 训练模型

```bash
python main.py \
  --mode train \
  --data_dir data/ml-1m \
  --kg_dir data/kg \
  --llm_model meta-llama/Llama-2-7b-chat-hf \
  --device cuda
```

### 2. 评估模型

```bash
python main.py \
  --mode eval \
  --data_dir data/ml-1m \
  --output_dir checkpoints/kragrec_ml1m
```

### 3. Ablation Study

```bash
python run_ablation_study.py \
  --data_dir data/ml-1m \
  --output_dir results/ablation
```

### 4. 超参数调优

```bash
python tune_hyperparameters.py \
  --tune_all \
  --output_dir results/tuning
```

### 5. 完整对比实验

```bash
python run_experiments.py \
  --data_dir data/ml-1m \
  --output_dir results/experiments
```

---

## 📈 论文对应关系

| 论文章节 | 代码文件 | 说明 |
|---------|---------|------|
| Section 3.3 | `indexing/index_KG.py` | 分层知识子图索引 |
| Section 3.4 | `retrieval/retriever.py` | 自适应检索策略 |
| Section 3.5 | `retrieval/retrieve.py` | 知识子图检索 |
| Section 3.6 | `retrieval/retrieve.py` | 知识子图重排序 |
| Section 3.7 | `generation/generator.py` | 知识增强推荐生成 |
| Section 4.2 | `run_experiments.py` | 完整对比实验 (Table 1) |
| Section 4.3 | `run_ablation_study.py` | Ablation Study (Figure 3) |
| Section 4.5 | `tune_hyperparameters.py` | 参数分析 (Figure 4-6) |

---

## 🛠️ 配置说明

### 模型配置 (`configs/config.py`)

```python
# GNN 配置
gnn_num_layers: 3        # GNN 层数 (ML-1M: 3, ML-20M/Book: 4)
gnn_hidden_dim: 128      # 隐藏层维度
gnn_output_dim: 256      # 输出维度

# 检索配置
topk_nodes: 10           # 初始检索数量 K
topk_rerank: 5           # 重排序数量 N
popularity_threshold: 0.5 # 流行度阈值 p

# LLM 配置
llm_model_name: "meta-llama/Llama-2-7b-chat-hf"
llm_max_length: 2048
llm_temperature: 0.7
```

### 训练配置

```python
batch_size: 5
learning_rate: 1e-5
num_epochs: 3
```

---

## 📚 数据集

### MovieLens-1M
- **下载**: https://grouplens.org/datasets/movielens/1m/
- **统计**: 6,040 用户, 3,706 电影, 1M 评分

### MovieLens-20M
- **下载**: https://grouplens.org/datasets/movielens/20m/
- **统计**: 138,493 用户, 26,744 电影, 20M 评分

### Amazon Book
- **下载**: http://jmcauley.ucsd.edu/data/amazon/
- **统计**: 294,739 用户, 367,982 书籍, 10M 评分

### Freebase 知识图谱
- **来源**: https://developers.google.com/freebase
- **说明**: 使用电影相关子集，或通过 `download_data.py` 自动构建

---

## 🔧 常见问题

### 1. CUDA Out of Memory

**解决方案**: 使用 4-bit 量化
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
```

### 2. HuggingFace 下载慢

**解决方案**: 使用镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. PyG 安装失败

**解决方案**: 指定 CUDA 版本
```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

更多问题见 [REPRODUCTION_STEPS.md](REPRODUCTION_STEPS.md)

---

## 📖 引用

如果这个项目对你有帮助，请引用原论文：

```bibtex
@article{wang2025knowledge,
  title={Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation},
  author={Wang, Shijie and Fan, Wenqi and Feng, Yue and Ma, Xinyu and Wang, Shuaiqiang and Yin, Dawei},
  journal={arXiv preprint arXiv:2501.02226},
  year={2025}
}
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

- **作者主页**: https://sjay-wang.github.io/
- **论文**: https://arxiv.org/abs/2501.02226

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

感谢以下项目和资源：
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Transformers](https://huggingface.co/docs/transformers/)
- [SentenceTransformers](https://www.sbert.net/)
- [MovieLens](https://grouplens.org/datasets/movielens/)
- [Freebase](https://developers.google.com/freebase)

---

**⭐ 如果觉得有用，请给个 Star！**
