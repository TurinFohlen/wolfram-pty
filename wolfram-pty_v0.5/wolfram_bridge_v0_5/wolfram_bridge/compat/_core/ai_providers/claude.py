#!/usr/bin/env python3
# === BEGIN METADATA ===
# name: claude
# description: Anthropic Claude API 的 AI 提供商实现
# usage: 由 query_engine 动态加载，用于命令解释和文本生成
# version: 1.1.0
# author: TurinFohlen
# dependencies: requests, ai_providers.base
# tags: AI, claude, 提供商
# === END METADATA ===

import os
import requests
from .base import AIProvider

class ClaudeProvider(AIProvider):
    def __init__(self, api_key: str = None):
        """
        初始化Claude提供商
        
        Args:
            api_key: API密钥（优先使用传入的密钥，否则从环境变量读取）
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("需要提供 ANTHROPIC_API_KEY")
        self.api_endpoint = "https://api.anthropic.com/v1/messages"

    def generate(self, prompt: str, **kwargs) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_tokens', 500)
        }
        response = requests.post(self.api_endpoint, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()['content'][0]['text']

    def explain_command(self, cmd_name: str, cmd_info: dict) -> str:
        prompt = f"请详细解释命令 {cmd_name} 的用法和最佳实践。\n命令信息：{cmd_info}"
        return self.generate(prompt)
