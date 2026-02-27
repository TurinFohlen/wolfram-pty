"""
compat/perf.py — 性能工具兼容代理（numba / jax / scipy.sparse 等）
用法：
    from wolfram_bridge.compat import perf
    # 查询可用映射：
    # perf.search("compile")
"""
import sys
from ._proxy_base import LibraryProxy, search, list_mappings

sys.modules[__name__] = LibraryProxy("numba")
