"""
知识增强推荐生成模块
将检索到的知识子图融入 LLM，生成推荐结果
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)


@dataclass
class RecommendationResult:
    """推荐结果"""
    item_name: str
    item_id: int
    score: float
    reason: str
    knowledge_used: List[str]


class RecommendationGenerator:
    """
    知识增强推荐生成器
    
    对应论文模块5：知识增强推荐生成
    - 将检索到的知识子图序列化为文本
    - 构建知识增强的 prompt
    - 使用 LLM 生成推荐结果
    
    支持两种知识融合方式：
    1. 文本拼接：将知识序列化为文本拼接到 prompt
    2. 软提示：通过投影器将知识转换为软提示 token
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cuda",
        max_length: int = 2048,
        temperature: float = 0.7,
        use_8bit: bool = False,
        use_4bit: bool = False
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        load_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "torch_dtype": torch.float16,
            "device_map": "auto" if device == "cuda" else None,
        }
        
        if use_8bit:
            load_kwargs["load_in_8bit"] = True
        elif use_4bit:
            load_kwargs["load_in_4bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        
        if device == "cuda" and not use_8bit and not use_4bit:
            self.model = self.model.to(device)
        
        self.model.eval()
        
        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
    
    def build_prompt(
        self,
        user_history: List[str],
        retrieved_knowledge: List[str],
        num_recommendations: int = 5,
        task_type: str = "sequential"
    ) -> str:
        """
        构建知识增强的推荐 prompt
        
        Args:
            user_history: 用户历史交互物品列表
            retrieved_knowledge: 检索到的知识文本列表
            num_recommendations: 推荐数量
            task_type: 任务类型 (sequential, rating, explanation)
        
        Returns:
            构建好的 prompt
        """
        history_text = "\n".join([f"- {item}" for item in user_history[-10:]])
        
        knowledge_text = "\n".join(retrieved_knowledge[:5])
        
        if task_type == "sequential":
            prompt = f"""You are a recommendation system assistant. Based on the user's history and relevant knowledge, recommend {num_recommendations} items.

User's History:
{history_text}

Relevant Knowledge:
{knowledge_text}

Please recommend {num_recommendations} items that the user might be interested in. For each recommendation, provide:
1. Item name
2. Brief reason why this item is recommended

Format your response as:
1. [Item Name]: [Reason]
2. [Item Name]: [Reason]
...

Recommendations:"""
        
        elif task_type == "rating":
            prompt = f"""You are a recommendation system assistant. Based on the user's history and relevant knowledge, predict the user's rating for items.

User's History:
{history_text}

Relevant Knowledge:
{knowledge_text}

Please predict the user's rating (1-5 stars) for the target item and explain why.

Rating Prediction:"""
        
        else:
            prompt = f"""You are a recommendation system assistant. Based on the user's history and relevant knowledge, provide personalized recommendations.

User's History:
{history_text}

Relevant Knowledge:
{knowledge_text}

Please provide your recommendations:"""
        
        return prompt
    
    def build_chat_prompt(
        self,
        user_history: List[str],
        retrieved_knowledge: List[str],
        num_recommendations: int = 5
    ) -> str:
        """
        构建对话格式的 prompt (适用于 LLaMA-Chat 等模型)
        
        Args:
            user_history: 用户历史
            retrieved_knowledge: 检索到的知识
            num_recommendations: 推荐数量
        
        Returns:
            对话格式的 prompt
        """
        history_text = ", ".join(user_history[-10:])
        knowledge_text = "\n".join(retrieved_knowledge[:3])
        
        system_prompt = """You are an intelligent recommendation assistant. Your task is to provide personalized recommendations based on user history and relevant knowledge. Be concise and helpful."""
        
        user_prompt = f"""Based on my viewing history: {history_text}

And the following knowledge about related items:
{knowledge_text}

Please recommend {num_recommendations} items I might like. For each item, briefly explain why it's recommended."""
        
        if "llama" in self.model_name.lower() or "mistral" in self.model_name.lower():
            full_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        else:
            full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        
        return full_prompt
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256
    ) -> str:
        """
        生成推荐文本
        
        Args:
            prompt: 输入 prompt
            max_new_tokens: 最大生成 token 数
        
        Returns:
            生成的文本
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def generate_recommendations(
        self,
        user_history: List[str],
        retrieved_knowledge: List[str],
        num_recommendations: int = 5,
        use_chat_format: bool = True
    ) -> str:
        """
        生成推荐结果
        
        Args:
            user_history: 用户历史
            retrieved_knowledge: 检索到的知识
            num_recommendations: 推荐数量
            use_chat_format: 是否使用对话格式
        
        Returns:
            推荐文本
        """
        if use_chat_format:
            prompt = self.build_chat_prompt(
                user_history,
                retrieved_knowledge,
                num_recommendations
            )
        else:
            prompt = self.build_prompt(
                user_history,
                retrieved_knowledge,
                num_recommendations
            )
        
        return self.generate(prompt)
    
    def parse_recommendations(
        self,
        generated_text: str
    ) -> List[RecommendationResult]:
        """
        解析生成的推荐文本
        
        Args:
            generated_text: 生成的文本
        
        Returns:
            推荐结果列表
        """
        results = []
        lines = generated_text.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if ":" in line:
                parts = line.split(":", 1)
                item_name = parts[0].strip()
                
                import re
                item_name = re.sub(r'^\d+\.\s*', '', item_name)
                item_name = re.sub(r'^\d+\)\s*', '', item_name)
                item_name = re.sub(r'^[-*]\s*', '', item_name)
                
                reason = parts[1].strip() if len(parts) > 1 else ""
                
                results.append(RecommendationResult(
                    item_name=item_name,
                    item_id=-1,
                    score=0.0,
                    reason=reason,
                    knowledge_used=[]
                ))
        
        return results


class KnowledgeEnhancedGenerator(RecommendationGenerator):
    """
    知识增强生成器
    支持软提示知识融合
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cuda",
        knowledge_dim: int = 256,
        num_soft_tokens: int = 10,
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        
        self.knowledge_dim = knowledge_dim
        self.num_soft_tokens = num_soft_tokens
        self.llm_hidden_dim = self.model.config.hidden_size
        
        self.knowledge_proj = nn.Sequential(
            nn.Linear(knowledge_dim, self.llm_hidden_dim),
            nn.LayerNorm(self.llm_hidden_dim),
            nn.GELU(),
            nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim * num_soft_tokens)
        ).to(device)
    
    def encode_knowledge(
        self,
        knowledge_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        将知识嵌入编码为软提示 token
        
        Args:
            knowledge_embedding: 知识嵌入 [batch_size, knowledge_dim]
        
        Returns:
            软提示 token [batch_size, num_soft_tokens, llm_hidden_dim]
        """
        batch_size = knowledge_embedding.size(0)
        
        soft_tokens = self.knowledge_proj(knowledge_embedding)
        soft_tokens = soft_tokens.view(batch_size, self.num_soft_tokens, self.llm_hidden_dim)
        
        return soft_tokens
    
    def generate_with_soft_prompt(
        self,
        prompt: str,
        knowledge_embedding: torch.Tensor,
        max_new_tokens: int = 256
    ) -> str:
        """
        使用软提示生成
        
        Args:
            prompt: 文本 prompt
            knowledge_embedding: 知识嵌入
            max_new_tokens: 最大生成 token 数
        
        Returns:
            生成的文本
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - self.num_soft_tokens - max_new_tokens
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.model.get_input_embeddings()(input_ids)
        
        soft_tokens = self.encode_knowledge(knowledge_embedding)
        
        combined_embeddings = torch.cat([soft_tokens, text_embeddings], dim=1)
        
        soft_attention = torch.ones(
            knowledge_embedding.size(0),
            self.num_soft_tokens,
            device=self.device
        )
        combined_attention = torch.cat([soft_attention, attention_mask], dim=1)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
