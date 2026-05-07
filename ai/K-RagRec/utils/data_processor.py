"""
数据处理工具模块
"""

import os
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm


@dataclass
class UserItemInteraction:
    """用户-物品交互"""
    user_id: int
    item_id: int
    rating: float
    timestamp: int


class MovieLensProcessor:
    """
    MovieLens 数据集处理器
    
    支持 MovieLens-1M 和 MovieLens-20M
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.users: Dict[int, Dict] = {}
        self.items: Dict[int, Dict] = {}
        self.interactions: List[UserItemInteraction] = []
        self.user_item_dict: Dict[int, List[int]] = defaultdict(list)
    
    def load_ratings(self, file_name: str = "ratings.dat") -> List[UserItemInteraction]:
        """
        加载评分数据
        
        Args:
            file_name: 评分文件名
        
        Returns:
            交互列表
        """
        file_path = os.path.join(self.data_dir, file_name)
        
        with open(file_path, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) >= 4:
                    user_id = int(parts[0])
                    item_id = int(parts[1])
                    rating = float(parts[2])
                    timestamp = int(parts[3])
                    
                    interaction = UserItemInteraction(
                        user_id=user_id,
                        item_id=item_id,
                        rating=rating,
                        timestamp=timestamp
                    )
                    self.interactions.append(interaction)
                    self.user_item_dict[user_id].append(item_id)
        
        return self.interactions
    
    def load_movies(self, file_name: str = "movies.dat") -> Dict[int, Dict]:
        """
        加载电影元数据
        
        Args:
            file_name: 电影文件名
        
        Returns:
            电影字典 {item_id: {title, genres}}
        """
        file_path = os.path.join(self.data_dir, file_name)
        
        with open(file_path, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) >= 3:
                    item_id = int(parts[0])
                    title = parts[1]
                    genres = parts[2].split("|")
                    
                    year_match = re.search(r'\((\d{4})\)$', title)
                    year = int(year_match.group(1)) if year_match else None
                    
                    self.items[item_id] = {
                        "title": title,
                        "genres": genres,
                        "year": year
                    }
        
        return self.items
    
    def load_users(self, file_name: str = "users.dat") -> Dict[int, Dict]:
        """
        加载用户数据
        
        Args:
            file_name: 用户文件名
        
        Returns:
            用户字典
        """
        file_path = os.path.join(self.data_dir, file_name)
        
        if not os.path.exists(file_path):
            return self.users
        
        with open(file_path, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) >= 5:
                    user_id = int(parts[0])
                    self.users[user_id] = {
                        "gender": parts[1],
                        "age": int(parts[2]),
                        "occupation": int(parts[3]),
                        "zip_code": parts[4]
                    }
        
        return self.users
    
    def get_user_history(
        self,
        user_id: int,
        max_length: int = 20
    ) -> List[int]:
        """
        获取用户历史交互
        
        Args:
            user_id: 用户 ID
            max_length: 最大历史长度
        
        Returns:
            物品 ID 列表
        """
        interactions = [
            i for i in self.interactions
            if i.user_id == user_id
        ]
        interactions.sort(key=lambda x: x.timestamp)
        
        item_ids = [i.item_id for i in interactions]
        
        return item_ids[-max_length:]
    
    def get_item_names(self, item_ids: List[int]) -> List[str]:
        """
        获取物品名称列表
        
        Args:
            item_ids: 物品 ID 列表
        
        Returns:
            物品名称列表
        """
        names = []
        for item_id in item_ids:
            if item_id in self.items:
                names.append(self.items[item_id]["title"])
            else:
                names.append(f"Item_{item_id}")
        return names
    
    def split_by_time(
        self,
        train_ratio: float = 0.8
    ) -> Tuple[List[UserItemInteraction], List[UserItemInteraction]]:
        """
        按时间划分训练集和测试集
        
        Args:
            train_ratio: 训练集比例
        
        Returns:
            (训练交互, 测试交互)
        """
        sorted_interactions = sorted(
            self.interactions,
            key=lambda x: x.timestamp
        )
        
        split_idx = int(len(sorted_interactions) * train_ratio)
        
        train = sorted_interactions[:split_idx]
        test = sorted_interactions[split_idx:]
        
        return train, test
    
    def build_kg_triples(
        self,
        output_dir: str
    ):
        """
        从电影数据构建知识图谱三元组
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        entities = {}
        relations = {}
        triples = []
        
        relation_names = ["directed_by", "starring", "genre", "year"]
        for i, rel in enumerate(relation_names):
            relations[rel] = i
        
        entity_id = 0
        
        for item_id, item_info in self.items.items():
            if item_id not in entities:
                entities[item_id] = entity_id
                entity_id += 1
            
            movie_entity = entities[item_id]
            
            for genre in item_info.get("genres", []):
                if genre not in entities:
                    entities[genre] = entity_id
                    entity_id += 1
                genre_entity = entities[genre]
                triples.append((movie_entity, relations["genre"], genre_entity))
            
            if item_info.get("year"):
                year_str = str(item_info["year"])
                if year_str not in entities:
                    entities[year_str] = entity_id
                    entity_id += 1
                year_entity = entities[year_str]
                triples.append((movie_entity, relations["year"], year_entity))
        
        with open(os.path.join(output_dir, "entities.txt"), "w", encoding="utf-8") as f:
            for name, eid in entities.items():
                f.write(f"{eid}\t{name}\tentity\n")
        
        with open(os.path.join(output_dir, "relations.txt"), "w", encoding="utf-8") as f:
            for name, rid in relations.items():
                f.write(f"{rid}\t{name}\n")
        
        with open(os.path.join(output_dir, "triples.txt"), "w", encoding="utf-8") as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")
        
        print(f"构建知识图谱完成:")
        print(f"  - 实体数量: {len(entities)}")
        print(f"  - 关系数量: {len(relations)}")
        print(f"  - 三元组数量: {len(triples)}")
        
        return entities, relations, triples


def create_sample_data(output_dir: str):
    """
    创建示例数据用于测试
    
    Args:
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    entities = [
        (0, "Toy Story", "movie"),
        (1, "Jumanji", "movie"),
        (2, "The Matrix", "movie"),
        (3, "Inception", "movie"),
        (4, "Interstellar", "movie"),
        (5, "Animation", "genre"),
        (6, "Adventure", "genre"),
        (7, "Sci-Fi", "genre"),
        (8, "Action", "genre"),
        (9, "Christopher Nolan", "director"),
        (10, "Lana Wachowski", "director"),
        (11, "Tom Hanks", "actor"),
        (12, "Leonardo DiCaprio", "actor"),
        (13, "Matthew McConaughey", "actor"),
    ]
    
    relations = [
        (0, "has_genre"),
        (1, "directed_by"),
        (2, "starring"),
        (3, "similar_to"),
    ]
    
    triples = [
        (0, 0, 5),
        (0, 0, 6),
        (1, 0, 5),
        (1, 0, 6),
        (2, 0, 7),
        (2, 0, 8),
        (3, 0, 7),
        (3, 0, 8),
        (4, 0, 7),
        (2, 1, 10),
        (3, 1, 9),
        (4, 1, 9),
        (0, 2, 11),
        (3, 2, 12),
        (4, 2, 13),
        (2, 3, 3),
        (3, 3, 4),
    ]
    
    with open(os.path.join(output_dir, "entities.txt"), "w", encoding="utf-8") as f:
        for eid, name, etype in entities:
            f.write(f"{eid}\t{name}\t{etype}\n")
    
    with open(os.path.join(output_dir, "relations.txt"), "w", encoding="utf-8") as f:
        for rid, name in relations:
            f.write(f"{rid}\t{name}\n")
    
    with open(os.path.join(output_dir, "triples.txt"), "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")
    
    print(f"示例数据已创建: {output_dir}")
