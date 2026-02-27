"""
compat/matplotlib.py — Matplotlib / Seaborn 绘图代理
用法：
    from wolfram_bridge.compat import matplotlib as plt
    img = plt.pyplot.scatter([1,2,3], [4,5,6])
"""
import sys
from ._proxy_base import LibraryProxy
sys.modules[__name__] = LibraryProxy("matplotlib")
