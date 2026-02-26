import sys
sys.path.insert(0, '/root/wolfproject')
from wolfram_bridge import WolframKernel
import os
import json

k = WolframKernel()

# 设置输出目录（确保可写）
OUT_DIR = "/sdcard/wolfram_out"
os.makedirs(OUT_DIR, exist_ok=True)

print("=== 测试 evaluate_to_file (JSON) ===")
file1 = k.evaluate_to_file("2+2", fmt="json", out_dir=OUT_DIR)
print(f"结果文件1: {file1}")
with open(file1, 'r') as f:
    data = json.load(f)
    print(f"文件内容: {data}")

print("\n=== 测试 evaluate_to_file (TXT) ===")
file2 = k.evaluate_to_file("N[Pi, 20]", fmt="txt", out_dir=OUT_DIR)
print(f"结果文件2: {file2}")
with open(file2, 'r') as f:
    content = f.read()
    print(f"文件内容: {content}")

print("\n=== 测试 eval_array (自动清理) ===")
arr = k.eval_array("RandomReal[1, {3,3}]")
print("numpy 数组:\n", arr)

print("\n=== 测试 eval_image (自动清理) ===")
img = k.eval_image("MandelbrotSetPlot[{0.3,0.6}, {0.3+1.1I,0.6+1.1I}, ImageSize->400]")
img.show()  # 可能需要在有图形界面的环境才能显示，否则会报错，但文件已保存
print("图像已生成并自动清理临时文件")

print("\n✅ 所有测试完成。生成的文件保存在:", OUT_DIR)
