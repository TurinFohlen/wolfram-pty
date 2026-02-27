"""
compat/numpy_extra.py — NumPy 补充映射代理
涵盖：集合运算/类型检查/多项式/直方图/窗函数/位运算等
这些规则挂在 numpy.* 命名空间下，通过主 numpy proxy 访问即可；
此模块单独暴露供需要直接 import 的场景。
"""
import sys
from ._proxy_base import LibraryProxy
sys.modules[__name__] = LibraryProxy("numpy")
