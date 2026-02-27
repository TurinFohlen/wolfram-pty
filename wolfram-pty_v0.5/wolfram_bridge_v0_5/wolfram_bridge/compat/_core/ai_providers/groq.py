#!/usr/bin/env python3
# === BEGIN METADATA ===
# name: groq
# description: Groq API (Llama 3.3) 的 AI 提供商实现
# usage: 由 query_engine 动态加载，用于极速命令解释
# version: 1.2.0
# author: TurinFohlen
# dependencies: groq, ai_providers.base
# tags: AI, groq, llama, 提供商
# === END METADATA ===

import os
from groq import Groq
from .base import AIProvider

class GroqProvider(AIProvider):
    def __init__(self, api_key: str = None):
        # 直接保存密钥，不抛出异常（上层已保证非空）
        self.api_key = api_key
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.model = "llama-3.3-70b-versatile"

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.client:
            return "❌ Groq 客户端未初始化，请检查 API Key"
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2048)
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"❌ Groq 服务调用失败: {str(e)}"

    def explain_command(self, cmd_name: str, cmd_info: dict) -> str:
        prompt = f"""你是一位 Linux/Unix 命令行专家。请详细解释命令 `{cmd_name}` 的用法和最佳实践。

本地索引信息：
- **描述**：{cmd_info.get('description', '无')}
- **用法**：{cmd_info.get('usage', '无')}
- **标签**：{', '.join(cmd_info.get('tags', []))}

请提供以下内容（使用 Markdown 格式）：

1. **命令简介**：一句话说明它的主要作用。
2. **常见用法**：列举 5-8 个最常见的用法示例，每个示例包含命令行和简要说明。
3. **核心参数详解**：解释最常用的几个选项（如 -l, -a, -h 等），说明其作用和典型组合。
4. **实用技巧**：2-3 个提高效率的技巧或组合用法。
5. **相关命令**：列出与该命令相关的其他命令（如 tree, stat, find 等）。

请确保解释清晰、准确，并尽量覆盖日常使用场景。
"""
        return self.generate(prompt, max_tokens=2048, temperature=0.6)
