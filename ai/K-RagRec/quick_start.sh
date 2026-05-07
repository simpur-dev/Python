#!/bin/bash
# K-RagRec 快速启动脚本
# 一键完成环境检查、数据下载、模块测试

echo "============================================================"
echo "K-RagRec 快速启动"
echo "ACL 2025"
echo "============================================================"

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 步骤 1: 环境检查
echo -e "\n${YELLOW}[步骤 1/5] 检查环境...${NC}"
python check_env.py
if [ $? -ne 0 ]; then
    echo -e "${RED}环境检查失败，请先安装依赖${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 环境检查通过${NC}"

# 步骤 2: 下载数据
echo -e "\n${YELLOW}[步骤 2/5] 下载数据集...${NC}"
if [ -d "K-RagRec/data/ml-1m" ]; then
    echo -e "${GREEN}✓ 数据集已存在，跳过下载${NC}"
else
    python K-RagRec/download_data.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}数据下载失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 数据下载完成${NC}"
fi

# 步骤 3: 模块测试
echo -e "\n${YELLOW}[步骤 3/5] 运行模块测试...${NC}"
python K-RagRec/run_full_test.py
if [ $? -ne 0 ]; then
    echo -e "${RED}模块测试失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 模块测试通过${NC}"

# 步骤 4: Demo 运行
echo -e "\n${YELLOW}[步骤 4/5] 运行 Demo...${NC}"
python K-RagRec/main.py --mode demo --use_sample_data
if [ $? -ne 0 ]; then
    echo -e "${RED}Demo 运行失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Demo 运行成功${NC}"

# 步骤 5: 完成
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}快速启动完成!${NC}"
echo -e "${GREEN}============================================================${NC}"

echo -e "\n下一步操作:"
echo -e "  1. 训练模型: ${YELLOW}python K-RagRec/main.py --mode train${NC}"
echo -e "  2. 评估模型: ${YELLOW}python K-RagRec/main.py --mode eval${NC}"
echo -e "  3. Ablation Study: ${YELLOW}python K-RagRec/run_ablation_study.py${NC}"
echo -e "  4. 完整实验: ${YELLOW}python K-RagRec/run_experiments.py${NC}"

echo -e "\n查看详细文档: ${YELLOW}REPRODUCTION_STEPS.md${NC}"
