#!/usr/bin/env python3
import os
import time
import hashlib
from wolfram_bridge import WolframKernel, CACHE_DIR

def test_cache():
    print("="*50)
    print("开始测试缓存功能")
    print("="*50)

    k = WolframKernel()
    expr = "Range[1000]"          # 一个中等大小的数组
    fmt = "json"

    # 计算哈希，用于定位缓存文件
    expr_hash = hashlib.sha256(expr.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{expr_hash}.{fmt}")

    # 第一次调用（应触发实际计算）
    print("\n[1] 第一次调用 (应触发内核计算)")
    path1 = k.evaluate_to_file(expr, fmt=fmt)
    print(f"    结果文件: {path1}")
    print(f"    缓存文件是否存在? {os.path.exists(cache_path)}")

    # 短暂等待，确保文件写完毕
    time.sleep(1)

    # 第二次调用（应命中缓存）
    print("\n[2] 第二次调用 (应命中缓存，不触发内核)")
    path2 = k.evaluate_to_file(expr, fmt=fmt)
    print(f"    结果文件: {path2}")
    # 缓存文件应该还在
    print(f"    缓存文件是否存在? {os.path.exists(cache_path)}")

    # 清理缓存
    print("\n[3] 清除缓存")
    k.clear_cache()
    print(f"    缓存文件是否存在? {os.path.exists(cache_path)}")

    # 第三次调用（缓存已清，应重新计算）
    print("\n[4] 第三次调用 (缓存已清，应重新计算)")
    path3 = k.evaluate_to_file(expr, fmt=fmt)
    print(f"    结果文件: {path3}")
    print(f"    缓存文件是否存在? {os.path.exists(cache_path)}")

    print("\n✅ 测试完成。请观察控制台输出中是否出现内核计算日志（INFO 级别的 'Export[...]' 等）。")
    print("   如果第二次调用没有出现此类日志，说明缓存生效。")

if __name__ == "__main__":
    test_cache()
