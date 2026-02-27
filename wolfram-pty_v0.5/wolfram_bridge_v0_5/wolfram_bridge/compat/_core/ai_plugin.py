"""
compat/_core/ai_plugin.py
-------------------------
AI 插件：YAML 表找不到映射时，AI 给出 Wolfram 函数名建议。
复用 dict-tree 的 ai_providers 动态加载机制。

环境变量：
  AI_PROVIDER=deepseek|claude|gemini|groq  （默认 deepseek）
  DEEPSEEK_API_KEY / ANTHROPIC_API_KEY 等
"""

import os, importlib, pkgutil, logging
from typing import Optional

log = logging.getLogger("wolfram_bridge.compat")

_PROVIDERS_PKG  = "wolfram_bridge.compat._core.ai_providers"
_HARDCODED_MAP  = {
    "deepseek": "DeepSeekProvider",
    "claude":   "ClaudeProvider",
    "gemini":   "GeminiProvider",
    "groq":     "GroqProvider",
}
_ENV_KEY_MAP = {
    "deepseek": "DEEPSEEK_API_KEY",
    "claude":   "ANTHROPIC_API_KEY",
    "gemini":   "GEMINI_API_KEY",
    "groq":     "GROQ_API_KEY",
}


class AIPlugin:
    def __init__(self, api_key: str = None, provider_name: str = None):
        self._name     = (provider_name or os.getenv("AI_PROVIDER", "deepseek")).lower()
        self._api_key  = api_key
        self._provider = None

    def _ensure_provider(self) -> bool:
        if self._provider is not None:
            return True

        pkg = importlib.import_module(_PROVIDERS_PKG)
        dynamic = {m: m.capitalize() + "Provider"
                   for _, m, _ in pkgutil.iter_modules(pkg.__path__)
                   if m not in ("__init__", "base")}
        full_map = {**dynamic, **_HARDCODED_MAP}

        if self._name not in full_map:
            log.warning(f"未知提供商：{self._name}，可用：{list(full_map)}")
            return False

        if not self._api_key:
            env = _ENV_KEY_MAP.get(self._name, self._name.upper() + "_API_KEY")
            self._api_key = os.getenv(env)
        if not self._api_key:
            log.warning(f"未找到 {self._name} API Key，AI 插件已禁用")
            return False

        try:
            mod = importlib.import_module(f"{_PROVIDERS_PKG}.{self._name}")
            self._provider = getattr(mod, full_map[self._name])(api_key=self._api_key)
            log.info(f"AI 插件就绪：{self._name}")
            return True
        except Exception as e:
            log.warning(f"加载提供商失败：{e}")
            return False

    def suggest_mapping(self, query: str) -> Optional[str]:
        if not self._ensure_provider():
            return None
        try:
            prompt = (
                f"你是 Wolfram Language 专家。用户需要：{query}\n"
                f"只回答对应的 Wolfram 函数名，不要解释，不要括号，不要参数。"
            )
            return self._provider.generate(prompt, max_tokens=50)
        except Exception as e:
            log.warning(f"AI 建议失败：{e}")
            return None
