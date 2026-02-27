#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from compat import numpy as np
import json
import os

def load_json_file(path):
    """读取 JSON 文件并返回解析后的数据"""
    with open(path, 'r') as f:
        return json.load(f)

print("🚀 测试真实 Wolfram 内核调用（手动解析文件）")
print("="*50)

# 1. 创建数组（应该返回文件路径）
a_path = np.array([1, 2, 3, 4])
print(f"np.array 返回路径: {a_path}")
if os.path.exists(a_path):
    a = load_json_file(a_path)
    print(f"解析后的 a = {a}")
else:
    print("❌ 文件不存在")

# 2. FFT 计算
b_path = np.fft.fft([1, 2, 3, 4])
print(f"np.fft.fft 返回路径: {b_path}")
if os.path.exists(b_path):
    b = load_json_file(b_path)
    print(f"解析后的 b = {b}")
else:
    print("❌ 文件不存在")

# 3. 行列式（可能也是文件路径）
c_result = np.linalg.det([[1, 2], [3, 4]])
print(f"np.linalg.det 返回: {c_result}")
if isinstance(c_result, str) and os.path.exists(c_result):
    c = load_json_file(c_result)
    print(f"解析后的 c = {c}")
else:
    print(f"直接结果: {c_result}")

print("="*50)
print("测试完成")
