"""wolfram_bridge 包（懒加载核心，避免在没有 pexpect 的环境爆炸）"""

def __getattr__(name):
    if name in ("WolframKernel", "WolframPipeline"):
        from .wolfram_bridge import WolframKernel, WolframPipeline
        import sys
        sys.modules[__name__].WolframKernel  = WolframKernel
        sys.modules[__name__].WolframPipeline = WolframPipeline
        return locals()[name]
    raise AttributeError(f"module 'wolfram_bridge' has no attribute {name!r}")
