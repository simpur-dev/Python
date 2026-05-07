"""
数据集下载脚本
自动下载 MovieLens-1M 和构建知识图谱
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

def download_file(url, dest_path):
    """下载文件并显示进度"""
    print(f"正在下载: {url}")
    print(f"保存到: {dest_path}")
    
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r进度: {percent}% [{count * block_size}/{total_size} bytes]")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, dest_path, reporthook)
    print("\n下载完成!")

def download_movielens_1m(data_dir="data"):
    """下载 MovieLens-1M 数据集"""
    print("\n" + "="*60)
    print("下载 MovieLens-1M 数据集")
    print("="*60)
    
    os.makedirs(data_dir, exist_ok=True)
    
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(data_dir, "ml-1m.zip")
    extract_path = os.path.join(data_dir, "ml-1m")
    
    # 检查是否已存在
    if os.path.exists(extract_path):
        print(f"数据集已存在: {extract_path}")
        return extract_path
    
    # 下载
    if not os.path.exists(zip_path):
        download_file(url, zip_path)
    else:
        print(f"压缩包已存在: {zip_path}")
    
    # 解压
    print("\n正在解压...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print(f"解压完成: {extract_path}")
    
    # 删除压缩包
    os.remove(zip_path)
    print("已删除压缩包")
    
    return extract_path

def build_knowledge_graph(ml_dir, kg_dir="data/kg"):
    """从 MovieLens 数据构建知识图谱"""
    print("\n" + "="*60)
    print("构建知识图谱")
    print("="*60)
    
    os.makedirs(kg_dir, exist_ok=True)
    
    # 检查是否已存在
    if os.path.exists(os.path.join(kg_dir, "triples.txt")):
        print(f"知识图谱已存在: {kg_dir}")
        return kg_dir
    
    try:
        from utils.data_processor import MovieLensProcessor
        
        processor = MovieLensProcessor(ml_dir)
        print("\n[1/3] 加载评分数据...")
        processor.load_ratings()
        print(f"  加载了 {len(processor.interactions)} 条交互记录")
        
        print("\n[2/3] 加载电影元数据...")
        processor.load_movies()
        print(f"  加载了 {len(processor.items)} 部电影")
        
        print("\n[3/3] 构建知识图谱三元组...")
        entities, relations, triples = processor.build_kg_triples(kg_dir)
        
        print(f"\n知识图谱构建完成:")
        print(f"  实体数量: {len(entities)}")
        print(f"  关系数量: {len(relations)}")
        print(f"  三元组数量: {len(triples)}")
        print(f"  保存位置: {kg_dir}")
        
        return kg_dir
        
    except ImportError:
        print("\n警告: 无法导入 MovieLensProcessor")
        print("使用简化版知识图谱...")
        from utils.data_processor import create_sample_data
        create_sample_data(kg_dir)
        return kg_dir

def verify_data(ml_dir, kg_dir):
    """验证数据完整性"""
    print("\n" + "="*60)
    print("验证数据完整性")
    print("="*60)
    
    # 检查 MovieLens 文件
    ml_files = ["ratings.dat", "movies.dat", "users.dat"]
    print("\nMovieLens 文件:")
    for f in ml_files:
        path = os.path.join(ml_dir, f)
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {f}")
    
    # 检查知识图谱文件
    kg_files = ["entities.txt", "relations.txt", "triples.txt"]
    print("\n知识图谱文件:")
    for f in kg_files:
        path = os.path.join(kg_dir, f)
        exists = "✓" if os.path.exists(path) else "✗"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                lines = len(file.readlines())
            print(f"  {exists} {f} ({lines} 行)")
        else:
            print(f"  {exists} {f}")
    
    print("\n数据验证完成!")

def main():
    print("="*60)
    print("K-RagRec 数据准备脚本")
    print("="*60)
    
    # 设置路径
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    # 下载 MovieLens-1M
    ml_dir = download_movielens_1m(str(data_dir))
    
    # 构建知识图谱
    kg_dir = build_knowledge_graph(ml_dir, str(data_dir / "kg"))
    
    # 验证数据
    verify_data(ml_dir, kg_dir)
    
    print("\n" + "="*60)
    print("数据准备完成! 可以开始训练了")
    print("="*60)
    print("\n下一步:")
    print("  1. 运行模块测试: python run_full_test.py")
    print("  2. 运行 Demo: python main.py --mode demo --use_sample_data")
    print("  3. 开始训练: python main.py --mode train")

if __name__ == "__main__":
    main()
