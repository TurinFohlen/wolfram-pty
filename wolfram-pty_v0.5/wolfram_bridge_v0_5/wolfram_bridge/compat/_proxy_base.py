"""
compat/_proxy_base.py
---------------------
通用代理基础模块：把 Python 函数调用路由到 Wolfram Kernel。
numpy.py / torch.py / sympy.py 等都继承自此，只需声明自己的 root 命名空间。
"""

import os
import sys
import logging
from pathlib import Path
from ._state import _state

log = logging.getLogger("wolfram_bridge.compat")

_DEFAULT_MAPPINGS = str(Path(__file__).parent / "mappings")


def _get_resolver():
    if _state["resolver"] is not None:
        return _state["resolver"]
    from ._core.metadata  import MetadataRepository
    from ._core.resolver  import ResolutionEngine
    from ._core.ai_plugin import AIPlugin

    mappings_dir = os.environ.get("WOLFRAM_MAPPINGS_DIR", _DEFAULT_MAPPINGS)
    repo = MetadataRepository(mappings_dir)
    ai   = AIPlugin() if os.environ.get("WOLFRAM_AI_PLUGIN") else None
    _state["resolver"] = ResolutionEngine(repo, ai)
    log.info(f"兼容层初始化，映射目录：{mappings_dir}，共 {len(repo.all_rules)} 条规则")
    return _state["resolver"]


def _get_kernel():
    if _state["kernel"] is not None:
        return _state["kernel"]
    try:
        from wolfram_bridge.wolfram_bridge import WolframKernel
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "wfb_core", Path(__file__).parent.parent / "wolfram_bridge.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        WolframKernel = mod.WolframKernel
    _state["kernel"] = WolframKernel()
    return _state["kernel"]


class _WolframCallable:
    """封装单个 Wolfram 函数调用。"""
    __slots__ = ("_path",)

    def __init__(self, path: str):
        self._path = path

    def __call__(self, *args, **kwargs):
        resolver       = _get_resolver()
        rule           = resolver.resolve(self._path, args=args, kwargs=kwargs)
        if rule is None:
            hints = resolver.candidates_for_hint(self._path)
            raise AttributeError(
                f"未找到 '{self._path}' 的 Wolfram 映射。"
                + (f"\n候选：{[r['python_path'] for r in hints[:5]]}" if hints else "")
            )
        kernel         = _get_kernel()
        expr, tmp_path = resolver.build_wl_expr(rule, args, kwargs)
        try:
            raw = kernel.evaluate(expr)
        except Exception as e:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise RuntimeError(f"内核执行失败 [{self._path}]：{e}") from e

        from ._core.converters import convert_output
        oc = rule.get("output_converter", "from_wl_passthrough")
        if tmp_path:
            try:
                result = convert_output(tmp_path, oc)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            result = convert_output(raw, oc)
        return result

    def __repr__(self):
        return f"<WolframCallable '{self._path}'>"


class LibraryProxy:
    """
    通用命名空间代理。
    - 精确路径匹配 → _WolframCallable
    - 其余 → 递归 LibraryProxy（支持 np.fft.fft 这样的链式访问）
    """
    def __init__(self, path: str):
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        new_path = f"{object.__getattribute__(self, '_path')}.{name}"
        if _get_resolver()._repo.get_rule(new_path) is not None:
            return _WolframCallable(new_path)
        return LibraryProxy(new_path)

    def __call__(self, *args, **kwargs):
        return _WolframCallable(
            object.__getattribute__(self, "_path"))(*args, **kwargs)

    def __repr__(self):
        return f"<WolframProxy '{object.__getattribute__(self, '_path')}'>"


# ── 便利函数 ──────────────────────────────────────────────────────
def list_mappings():
    """列出所有已加载的映射规则。"""
    return _get_resolver()._repo.all_rules

def search(query: str):
    """按关键词/标签/路径前缀搜索映射。"""
    return _get_resolver()._repo.search_rules(query)

def reload_mappings(directory: str = None):
    """热重载映射目录。"""
    _state["resolver"] = None
    if directory:
        os.environ["WOLFRAM_MAPPINGS_DIR"] = directory
    _get_resolver()

def inject_kernel(kernel):
    """测试用：注入 MockKernel。"""
    _state["kernel"] = kernel
