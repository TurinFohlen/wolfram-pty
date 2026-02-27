"""
compat/_core/resolver.py
------------------------
查找引擎：三级解析策略
  1. 精确匹配（Trie O(depth)）
  2. 模糊匹配（标签 + 关键词倒排索引）
  3. AI 兜底（精确+模糊均失败时）
"""

import logging
from typing import Optional, Dict, List

log = logging.getLogger("wolfram_bridge.compat")


class ResolutionEngine:
    _instance = None

    @classmethod
    def get_instance(cls, metadata_repo=None, ai_plugin=None):
        if cls._instance is None:
            if metadata_repo is None:
                raise RuntimeError("首次初始化必须传入 metadata_repo")
            cls._instance = cls(metadata_repo, ai_plugin)
        return cls._instance

    def __init__(self, metadata_repo, ai_plugin=None):
        self._repo      = metadata_repo
        self._ai_plugin = ai_plugin
        self._cache: Dict[str, Optional[Dict]] = {}

    def resolve(self, python_path: str, args=(), kwargs=None, use_ai: bool = True) -> Optional[Dict]:
        if python_path in self._cache:
            return self._cache[python_path]

        rule = self._repo.get_rule(python_path)
        if rule:
            self._cache[python_path] = rule
            return rule

        candidates = self._repo.search_rules(python_path)
        if candidates:
            return candidates[0]

        if use_ai and self._ai_plugin:
            suggestion = self._ai_suggest(python_path)
            if suggestion:
                return suggestion

        return None

    def search(self, query: str) -> List[Dict]:
        return self._repo.search_rules(query)

    def _ai_suggest(self, python_path: str) -> Optional[Dict]:
        try:
            suggestion = self._ai_plugin.suggest_mapping(
                f"Python 函数 {python_path} 对应的 Wolfram 函数名是？只回答函数名。"
            )
            if suggestion:
                return {
                    "python_path":      python_path,
                    "wolfram_function": suggestion.strip(),
                    "input_converter":  "to_wl_list",
                    "output_converter": "from_wl_json",
                    "tags":             [],
                    "description":      f"AI 建议：{suggestion}",
                    "match_type":       "ai_suggestion",
                }
        except Exception as e:
            log.warning(f"AI 兜底失败：{e}")
        return None

    def set_ai_plugin(self, plugin):
        self._ai_plugin = plugin

    def clear_cache(self):
        self._cache.clear()

    def candidates_for_hint(self, python_path: str) -> List[Dict]:
        """按路径前缀或关键词返回候选映射（用于 hint/日志显示）"""
        return self._repo.search_rules(python_path)

    def build_wl_expr(self, rule: Dict, args: tuple, kwargs: dict) -> tuple:
        """
        把规则 + 实际参数组装成 Wolfram 表达式字符串。

        - 标量输出（from_wl_scalar / from_wl_passthrough）：
            返回 (core_expr, None)
        - 其他输出（数组 / 图像 / JSON）：
            创建临时文件，返回 (Export["tmppath", core_expr, "JSON"], tmp_path)
            Wolfram 写完文件，Python 读取，OS 级原子性保证。
        """
        import tempfile, os
        from . import converters as _cv

        wf  = rule.get("wolfram_function", "Identity")
        oc  = rule.get("output_converter", "from_wl_passthrough")

        # ── 输入转换 ──────────────────────────────────────────────
        ics_names = rule.get("input_converters")   # 每参数各自的转换器
        ic_name   = rule.get("input_converter", "to_wl_list")

        if ics_names:
            wl_parts = []
            for arg, cn in zip(args, ics_names):
                conv = getattr(_cv, cn, _cv.to_wl_list)
                wl_parts.append(conv(arg))
            # 剩余多余的参数用默认转换器
            default_conv = getattr(_cv, ic_name, _cv.to_wl_list)
            for arg in args[len(ics_names):]:
                wl_parts.append(default_conv(arg))
        else:
            conv = getattr(_cv, ic_name, _cv.to_wl_list)
            if len(args) == 0:
                wl_parts = []
            elif len(args) == 1:
                wl_parts = [conv(args[0])]
            else:
                wl_parts = [conv(a) for a in args]

        if kwargs:
            kw_conv = getattr(_cv, ic_name, _cv.to_wl_list)
            for k, v in kwargs.items():
                wl_parts.append(f"{k} -> {kw_conv(v)}")

        core_expr = f"{wf}[{', '.join(wl_parts)}]" if wl_parts else wf

        # ── 文件模式 vs 标量模式 ──────────────────────────────────
        _SCALAR_CONVERTERS = {"from_wl_scalar", "from_wl_passthrough"}
        if oc in _SCALAR_CONVERTERS:
            return core_expr, None

        # 图像用 PNG，其余用 JSON
        if oc == "from_wl_image":
            fmt = "PNG"
            suffix = ".png"
        else:
            fmt = "JSON"
            suffix = ".json"

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="wlb_")
        os.close(tmp_fd)
        # WL 路径需要正斜杠
        wl_path = tmp_path.replace("\\", "/")
        export_expr = f'Export["{wl_path}", N[{core_expr}], "{fmt}"]'
        return export_expr, tmp_path
