# K-RagRec 完整复现指南

基于论文：[Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation](https://arxiv.org/abs/2501.02226)  
**ACL 2025 Main Conference**

## 📌 重要说明

**论文中未提供官方 GitHub 仓库链接**，但你当前的代码结构已经包含了论文的所有核心模块。本指南将帮助你：
1. 验证现有代码的完整性
2. 准备数据集
3. 运行完整实验
4. 复现论文结果

---

## 📊 论文核心信息

### 作者团队
- **第一作者**: Shijie Wang (香港理工大学)
- **通讯作者**: Wenqi Fan
- **合作单位**: 香港理工大学、伯明翰大学、香港城市大学、百度

### 实验配置（论文 Section 4.1.4）
- **硬件**: 2 × NVIDIA A6000-48GB GPUs
- **骨干 LLM**: LLama-2-7B, LLama-3-8B, QWEN2-7B
- **编码器**: SentenceBERT
- **GNN**: 3层 Graph Transformer (ML-1M), 4层 (ML-20M, Amazon Book)
- **流行度阈值 p**: 50%
- **检索数量 K**: 3
- **重排序数量 N**: 5

### 数据集
| 数据集 | 用户数 | 物品数 | 交互数 | KG 实体数 | KG 三元组数 |
|--------|--------|--------|--------|-----------|-------------|
| MovieLens-1M | 6,040 | 3,706 | 1M | 182,011 | 1,241,995 |
| MovieLens-20M | 138,493 | 26,744 | 20M | 182,011 | 1,241,995 |
| Amazon Book | 294,739 | 367,982 | 10M | 88,572 | 2,557,746 |

### 期望指标（论文 Table 1）
**MovieLens-1M + LLama-2-7B + Prompt Tuning:**
- ACC: 0.435
- Recall@3: 0.725
- Recall@5: 0.831

---

## 🚀 第 0 步：环境检查

### 1. 检查当前环境

```bash
python check_env.py
```

**期望输出：**
```
PyTorch: 2.x.x
Transformers: 4.35+
SentenceTransformers: 2.2+
PyG: 2.4+
FAISS: installed
CUDA available: True
GPU: NVIDIA ...
```

### 2. 安装缺失依赖

```bash
cd K-RagRec
pip install -r requirements.txt
```

**如果 PyG 安装失败：**
```bash
# 根据你的 CUDA 版本选择（查看：nvcc --version）
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric
```

---

## 📦 第 1 步：准备数据集

### 方案 A：使用示例数据（快速测试）

```bash
python K-RagRec/main.py --mode demo --use_sample_data
```

这会自动创建示例知识图谱，验证框架是否正常工作。

### 方案 B：下载真实数据集

#### 1. MovieLens-1M

```bash
# 下载数据集
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip -d K-RagRec/data/

# 或使用 PowerShell
Invoke-WebRequest -Uri "https://files.grouplens.org/datasets/movielens/ml-1m.zip" -OutFile "ml-1m.zip"
Expand-Archive -Path "ml-1m.zip" -DestinationPath "K-RagRec/data/"
```

#### 2. Freebase 知识图谱

论文使用 Freebase 的电影相关子集。你需要：

**选项 1：从 Freebase 构建（复杂）**
```python
# 下载 Freebase dump（约 30GB）
# 过滤电影相关三元组
# 参考：https://developers.google.com/freebase
```

**选项 2：使用预处理脚本（推荐）**
```python
from K_RagRec.utils.data_processor import MovieLensProcessor

processor = MovieLensProcessor("K-RagRec/data/ml-1m")
processor.load_ratings()
processor.load_movies()
processor.build_kg_triples("K-RagRec/data/kg")
```

这会从 MovieLens 的电影元数据构建简化版知识图谱。

---

## 🧪 第 2 步：运行模块测试

### 测试所有模块（无需 LLM）

```bash
python K-RagRec/run_full_test.py
```

**期望输出：**
```
============================================================
K-RagRec 完整版测试
ACL 2025
============================================================

设备: cuda
GPU: NVIDIA ...

[测试模块1: 分层知识子图语义索引]
  嵌入维度: torch.Size([21, 384])
  各层嵌入: torch.Size([21, 256])

[测试模块2: 自适应检索策略]
  检索效率提升: 50.0%

[测试模块3&4: 知识子图检索与重排序]
  检索结果: [2, 3, 4, 10, 7]
  重排序结果: [2, 3, 4]

[测试模块5: 知识增强推荐生成 (模拟)]
  推荐结果: 1. The Matrix (movie)
            2. Inception (movie)
            3. Interstellar (movie)

============================================================
所有模块测试完成!
============================================================
```

---

## 🤖 第 3 步：接入真实 LLM

### 方案 1：使用 LLama-2-7B（需要 GPU）

#### 1. 获取 HuggingFace Token

访问 https://huggingface.co/settings/tokens 创建 token

```bash
# 设置环境变量
export HF_TOKEN="your_token_here"

# Windows PowerShell
$env:HF_TOKEN="your_token_here"
```

#### 2. 申请 LLama-2 访问权限

访问 https://huggingface.co/meta-llama/Llama-2-7b-chat-hf 申请访问

#### 3. 运行 Demo

```bash
python K-RagRec/main.py \
  --mode demo \
  --llm_model meta-llama/Llama-2-7b-chat-hf \
  --device cuda \
  --kg_dir data/kg
```

### 方案 2：使用量化模型（8GB 显存可用）

修改 `K-RagRec/generation/generator.py`：

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 方案 3：使用 OpenAI API（无需本地 GPU）

在 `K-RagRec/generation/generator.py` 添加：

```python
import openai

class OpenAIGenerator(RecommendationGenerator):
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

---

## 📈 第 4 步：训练和评估

### 1. 训练模型（Prompt Tuning）

```bash
python K-RagRec/main.py \
  --mode train \
  --data_dir data/ml-1m \
  --kg_dir data/kg \
  --llm_model meta-llama/Llama-2-7b-chat-hf \
  --output_dir checkpoints/kragrec_ml1m \
  --device cuda
```

**训练参数（论文配置）：**
- Batch size: 5
- Epochs: 3
- Learning rate: 1e-5
- GNN layers: 3 (ML-1M), 4 (ML-20M, Amazon Book)
- GNN hidden dim: 1024

### 2. 评估模型

```bash
python K-RagRec/main.py \
  --mode eval \
  --data_dir data/ml-1m \
  --kg_dir data/kg \
  --llm_model meta-llama/Llama-2-7b-chat-hf \
  --output_dir checkpoints/kragrec_ml1m \
  --device cuda
```

**评估指标：**
- Accuracy (ACC)
- Recall@3
- Recall@5

### 3. 期望结果对比

| 方法 | ACC | Recall@3 | Recall@5 |
|------|-----|----------|----------|
| 论文报告 (ML-1M) | 0.435 | 0.725 | 0.831 |
| 你的结果 | ? | ? | ? |

---

## 🔬 第 5 步：Ablation Study

### 1. 去掉 GNN 模块

修改 `K-RagRec/configs/config.py`：

```python
# 在 retrieve_and_rerank 中设置
use_gnn_for_retrieve=False
use_gnn_for_rerank=False
```

**期望结果：** ACC 下降约 37%（论文 Figure 3）

### 2. 去掉自适应检索

```python
# 在 config.py 中设置
use_adaptive_retrieval=False
popularity_threshold=1.0  # 检索所有物品
```

**期望结果：** 推理时间显著增加

### 3. 去掉重排序

```python
# 在 retrieve_and_rerank 中设置
topk_rerank = topk_retrieve  # 不进行重排序
```

### 4. 运行 Ablation 实验

```bash
python K-RagRec/run_ablation_study.py \
  --data_dir data/ml-1m \
  --output_dir results/ablation
```

---

## 🎯 第 6 步：扩展实验

### 1. 换数据集

#### Amazon Book

```bash
# 下载数据集
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz

# 运行实验
python K-RagRec/main.py \
  --mode train \
  --data_dir data/amazon-book \
  --kg_dir data/kg_book \
  --llm_model meta-llama/Llama-2-7b-chat-hf
```

### 2. 换 GNN 架构

在 `K-RagRec/model/gnn_encoder.py` 中尝试：
- GCN → GAT (Graph Attention)
- GCN → GraphSAGE
- GCN → GIN (Graph Isomorphism Network)

论文 Table 9 显示不同 GNN 性能接近。

### 3. 调整超参数

```python
# 在 config.py 中调整
gnn_num_layers: 3 → 4, 5  # 论文 Table 10
topk_nodes: 10 → 5, 15     # 论文 Figure 5
topk_rerank: 5 → 3, 7      # 论文 Figure 6
popularity_threshold: 0.5 → 0.3, 0.7  # 论文 Figure 4
```

---

## 🐛 常见问题

### 1. CUDA Out of Memory

**解决方案：**
```python
# 使用 4-bit 量化
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# 或减小 batch size
batch_size = 1

# 或使用 CPU
--device cpu
```

### 2. HuggingFace 下载慢

**解决方案：**
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型到本地
--llm_model /path/to/local/llama-2-7b-chat-hf
```

### 3. PyG 安装失败

**解决方案：**
```bash
# 使用 conda
conda install pytorch-geometric -c pyg

# 或指定 CUDA 版本
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 4. 找不到 Freebase 数据

**解决方案：**
使用简化版知识图谱构建脚本（见第 1 步方案 B 选项 2）

---

## 📝 复现检查清单

- [ ] 环境配置完成（PyTorch, PyG, Transformers）
- [ ] 数据集下载（MovieLens-1M）
- [ ] 知识图谱构建（Freebase 或简化版）
- [ ] 模块测试通过（run_full_test.py）
- [ ] LLM 接入成功（Demo 模式）
- [ ] 训练完成（3 epochs）
- [ ] 评估指标接近论文报告值
- [ ] Ablation Study 完成
- [ ] 扩展实验（可选）

---

## 📚 参考资源

### 论文相关
- 论文 PDF: https://arxiv.org/abs/2501.02226
- 论文 HTML: https://arxiv.org/html/2501.02226v2
- 作者主页: https://sjay-wang.github.io/

### 数据集
- MovieLens: https://grouplens.org/datasets/movielens/
- Amazon Reviews: http://jmcauley.ucsd.edu/data/amazon/
- Freebase: https://developers.google.com/freebase

### 技术文档
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Transformers: https://huggingface.co/docs/transformers/
- SentenceTransformers: https://www.sbert.net/

---

## 🎓 下一步创新方向

### 1. 模型改进
- 引入多模态知识（图片、视频特征）
- 使用更强的 GNN（如 UniMP, GraphGPS）
- 动态调整检索策略（强化学习）

### 2. 应用扩展
- 跨域推荐（电影 → 音乐）
- 实时推荐（流式数据）
- 可解释性增强（可视化知识路径）

### 3. 效率优化
- 知识图谱压缩
- 检索加速（近似最近邻）
- 模型蒸馏（7B → 1B）

---

## 📧 联系方式

如果遇到问题，可以：
1. 查看论文附录（详细实验设置）
2. 参考作者主页（可能有后续更新）
3. 在相关论坛提问（如 Reddit r/MachineLearning）

**祝复现顺利！🎉**
