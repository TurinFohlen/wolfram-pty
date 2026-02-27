#!/usr/bin/env python3
# === BEGIN METADATA ===
# name: base
# description: AI服务提供商抽象基类，定义统一接口
# usage: 作为其他Provider类的基类，不直接使用
# version: 1.0.0
# author: TurinFohlen
# dependencies: abc
# tags: AI, 抽象基类, 接口
# === END METADATA ===

from abc import ABC, abstractmethod

class AIProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """根据提示生成文本"""
        pass

    @abstractmethod
    def explain_command(self, cmd_name: str, cmd_info: dict) -> str:
        """解释命令"""
        pass
