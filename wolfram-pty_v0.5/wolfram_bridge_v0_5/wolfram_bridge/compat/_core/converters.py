"""
compat/_core/converters.py
--------------------------
参数转换器和结果转换器注册表。

约定：
  - 所有输入转换器接收 Python 对象，返回 Wolfram 表达式字符串
  - 所有输出转换器接收 Wolfram 输出字符串，返回 Python 对象
  - 转换器名称在 YAML 映射规则中引用
"""

import json
import re
import logging
from typing import Any

log = logging.getLogger("wolfram_bridge.compat")


# ═══════════════════════════════════════════════════════════════
#  输入转换器：Python → Wolfram 表达式字符串
# ═══════════════════════════════════════════════════════════════

def to_wl_list(value) -> str:
    """Python list / ndarray → Wolfram List {1,2,3}"""
    if hasattr(value, "tolist"):        # numpy array
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        inner = ", ".join(to_wl_list(v) if isinstance(v, (list, tuple))
                          else str(v) for v in value)
        return "{" + inner + "}"
    return str(value)


def to_wl_scalar(value) -> str:
    """Python 数值 → Wolfram 数值字符串"""
    return str(value)


def to_wl_matrix(value) -> str:
    """二维 list / ndarray → Wolfram 矩阵 {{...},{...}}"""
    if hasattr(value, "tolist"):
        value = value.tolist()
    rows = ", ".join(to_wl_list(row) for row in value)
    return "{" + rows + "}"


def to_wl_matrix_and_vector(value) -> str:
    """
    接收 (A, b) 元组，转换为两个 Wolfram 参数。
    用于 linalg.solve(A, b) → LinearSolve[A, b]
    """
    if isinstance(value, (list, tuple)) and len(value) == 2:
        A, b = value
        return to_wl_matrix(A) + ", " + to_wl_list(b)
    raise ValueError(f"to_wl_matrix_and_vector 期望 (A, b) 元组，得到 {type(value)}")


def to_wl_string(value) -> str:
    """Python str → Wolfram String（加引号）"""
    return f'"{value}"'


def to_wl_passthrough(value) -> str:
    """直接传递原始字符串（已经是 Wolfram 表达式）"""
    return str(value)


# ── 多参数转换辅助 ──────────────────────────────────────────────
def args_to_wl(args, kwargs, rule: dict) -> str:
    """
    根据映射规则把 Python 调用参数转换为 Wolfram 函数参数字符串。

    规则中可选字段：
      input_converter: str       单一转换器（所有参数打包传入）
      input_converters: [str]    每个参数各自的转换器名称
    """
    ic = rule.get("input_converter", "to_wl_passthrough")
    ics = rule.get("input_converters")   # 可选：每个参数单独指定

    if ics:
        # 多参数各自转换
        parts = []
        for i, (arg, conv_name) in enumerate(zip(args, ics)):
            conv = INPUT_CONVERTERS.get(conv_name, to_wl_passthrough)
            parts.append(conv(arg))
        return ", ".join(parts)
    else:
        # 单参数 or 全部打包
        conv = INPUT_CONVERTERS.get(ic, to_wl_passthrough)
        if len(args) == 1:
            return conv(args[0])
        elif len(args) > 1:
            return conv(list(args))
        else:
            return ""


# ═══════════════════════════════════════════════════════════════
#  输出转换器：Wolfram 输出字符串 → Python 对象
# ═══════════════════════════════════════════════════════════════

def from_wl_json(wl_result: str) -> Any:
    """
    文件模式（优先）：wl_result 是文件路径 → 读文件解析 JSON。
    回退模式：wl_result 是 JSON 字符串 → 直接解析。
    文件模式绕过所有终端噪声，是最可靠的通信方式。
    """
    import os
    s = wl_result.strip().strip('"')
    if os.path.exists(s):
        with open(s, encoding="utf-8") as f:
            return json.load(f)
    # 回退：字符串模式
    try:
        return json.loads(s.replace('\\"', '"'))
    except Exception:
        return wl_result


def from_wl_image(wl_result: str):
    """
    文件模式：wl_result 是 PNG/JPG 文件路径 → PIL Image。
    Wolfram 端：Export[tmpPath, expr, "PNG"]
    """
    import os
    s = wl_result.strip().strip('"')
    if not os.path.exists(s):
        raise FileNotFoundError(f"Wolfram 图像文件不存在：{s}")
    try:
        from PIL import Image
        img = Image.open(s)
        img.load()   # 读入内存，允许后续删除临时文件
        return img
    except ImportError:
        raise ImportError("需要安装 Pillow：pip install Pillow")


def from_wl_list(wl_result: str) -> list:
    """
    Wolfram {1, 2, 3} 格式 → Python list（不经过 JSON）。
    支持嵌套列表和复数。
    """
    s = wl_result.strip()
    # 替换 Wolfram 花括号为方括号
    s = s.replace("{", "[").replace("}", "]")
    # 处理 Wolfram 复数格式：a + b*I → [a, b]（简化处理）
    s = re.sub(r"(\S+)\s*\+\s*(\S+)\s*\*?\s*I",
               lambda m: f"complex({m.group(1)},{m.group(2)})", s)
    try:
        return eval(s, {"__builtins__": {}}, {"complex": complex})
    except Exception:
        return wl_result


def from_wl_scalar(wl_result: str) -> Any:
    """尝试解析为数值，失败返回字符串"""
    s = wl_result.strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def from_wl_numpy(wl_result: str):
    """文件模式 JSON → numpy array"""
    try:
        import numpy as np
        data = from_wl_json(wl_result)
        return np.array(data)
    except ImportError:
        return from_wl_list(wl_result)


def from_wl_csv(wl_result: str):
    """
    文件模式：wl_result 是 CSV 文件路径 → numpy array。
    Wolfram 端：Export[tmpPath, expr, "CSV"]
    """
    import os
    s = wl_result.strip().strip('"')
    if os.path.exists(s):
        try:
            import numpy as np
            return np.loadtxt(s, delimiter=",")
        except ImportError:
            import csv
            with open(s) as f:
                return list(csv.reader(f))
    return from_wl_list(wl_result)


def from_wl_passthrough(wl_result: str) -> str:
    """直接返回原始字符串"""
    return wl_result


# ═══════════════════════════════════════════════════════════════
#  注册表
# ═══════════════════════════════════════════════════════════════

INPUT_CONVERTERS = {
    "to_wl_list":               to_wl_list,
    "to_wl_scalar":             to_wl_scalar,
    "to_wl_matrix":             to_wl_matrix,
    "to_wl_matrix_and_vector":  to_wl_matrix_and_vector,
    "to_wl_string":             to_wl_string,
    "to_wl_passthrough":        to_wl_passthrough,
}

OUTPUT_CONVERTERS = {
    "from_wl_json":             from_wl_json,
    "from_wl_image":            from_wl_image,
    "from_wl_csv":              from_wl_csv,
    "from_wl_list":             from_wl_list,
    "from_wl_scalar":           from_wl_scalar,
    "from_wl_numpy":            from_wl_numpy,
    "from_wl_passthrough":      from_wl_passthrough,
}


def convert_input(value, converter_name: str) -> str:
    conv = INPUT_CONVERTERS.get(converter_name, to_wl_passthrough)
    return conv(value)


def convert_output(wl_result: str, converter_name: str) -> Any:
    conv = OUTPUT_CONVERTERS.get(converter_name, from_wl_passthrough)
    return conv(wl_result)


def register_input_converter(name: str, func):
    """允许用户注册自定义输入转换器"""
    INPUT_CONVERTERS[name] = func


def register_output_converter(name: str, func):
    """允许用户注册自定义输出转换器"""
    OUTPUT_CONVERTERS[name] = func
