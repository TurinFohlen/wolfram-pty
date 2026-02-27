"""
compat/_state.py
全局可变状态容器，让测试可以直接注入 mock 对象。
"""
_state = {"kernel": None, "resolver": None}
