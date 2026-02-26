"""
wolfram_bridge.py
-----------------
极简 Wolfram-Python 桥梁
"""

import threading
import queue
import time
import os
import logging
import pexpect
import uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wolfram_bridge")

WOLFRAM_EXEC = os.environ.get("WOLFRAM_EXEC", "math")
WOLFRAM_ARGS = []
PROMPT_RE   = r"In\[\d+\]:="
STARTUP_TIMEOUT  = 60
EVAL_TIMEOUT     = 120

class WolframKernel:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._initialized = False
                cls._instance = obj
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._child = None
        self._kernel_lock = threading.Lock()
        self._req_queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._serial_loop, daemon=True)

        self._start_kernel()
        self._worker_thread.start()
        log.info("WolframKernel 单例已初始化")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        log.info("关闭 Wolfram 内核...")
        self._req_queue.put(None)
        with self._kernel_lock:
            if self._child and self._child.isalive():
                try:
                    self._child.sendline("Quit[]")
                    self._child.expect(pexpect.EOF, timeout=5)
                except Exception:
                    self._child.terminate(force=True)
            self._child = None
        with WolframKernel._instance_lock:
            WolframKernel._instance = None
        log.info("内核已关闭")

    def _start_kernel(self):
        log.info(f"启动内核：{WOLFRAM_EXEC}")
        child = pexpect.spawn(
            WOLFRAM_EXEC,
            WOLFRAM_ARGS,
            encoding="utf-8",
            echo=False,
            timeout=STARTUP_TIMEOUT,
            env=os.environ
        )
        try:
            child.expect(PROMPT_RE, timeout=STARTUP_TIMEOUT)
        except pexpect.TIMEOUT:
            child.terminate(force=True)
            raise RuntimeError("Wolfram 内核启动超时")
        except pexpect.EOF:
            raise RuntimeError(f"内核进程意外退出：{child.before}")

        with self._kernel_lock:
            self._child = child
        log.info("✅ 内核就绪")

    def _restart_kernel(self):
        log.warning("检测到内核异常，尝试重启...")
        with self._kernel_lock:
            if self._child and self._child.isalive():
                self._child.terminate(force=True)
            self._child = None
        time.sleep(1)
        self._start_kernel()
        log.info("✅ 内核已重启")

    def _send_command(self, cmd: str) -> str:
        """发送命令，返回从发送后到下一个提示符前的全部输出（不过滤）。"""
        with self._kernel_lock:
            child = self._child
        if child is None or not child.isalive():
            raise RuntimeError("内核未运行")

        child.sendline(cmd)
        child.expect(PROMPT_RE, timeout=EVAL_TIMEOUT)
        raw = child.before
        return raw.strip()

    # ---------- 公开 API ----------
    def evaluate(self, expr: str) -> str:
        """执行表达式，返回原始输出（可能包含回显）。"""
        return self._send_command(f'Print[{expr}]')

    def evaluate_to_file(self, expr: str, fmt: str = "json", out_dir: str = "/sdcard/wolfram_out") -> str:
        """
        执行表达式，将结果导出到文件，返回文件路径。
        自动创建目录，并等待文件生成（最多10秒）。
        格式映射表（后缀 -> Wolfram 格式名）：
            txt  → Text
            json → JSON
            csv  → CSV
            tsv  → TSV
            png  → PNG
            jpg/jpeg → JPEG
            gif  → GIF
            bmp  → BMP
            pdf  → PDF
            svg  → SVG
            eps  → EPS
            wdx  → WDX
            tex  → TeX
            table → Table
            list → List
            mx   → MX
        """
        # 格式名称映射（小写后缀 -> Wolfram 格式名）
        fmt_map = {
            "txt": "Text",
            "text": "Text",
            "json": "JSON",
            "csv": "CSV",
            "tsv": "TSV",
            "table": "Table",
            "list": "List",
            "wdx": "WDX",
            "png": "PNG",
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "gif": "GIF",
            "bmp": "BMP",
            "tif": "TIFF",
            "tiff": "TIFF",
            "pdf": "PDF",
            "svg": "SVG",
            "eps": "EPS",
            "tex": "TeX",
            "mx": "MX",
        }
        wl_format = fmt_map.get(fmt.lower())
        if wl_format is None:
            raise ValueError(
                f"不支持的格式 '{fmt}'。支持的格式有：{list(fmt_map.keys())}"
            )

        os.makedirs(out_dir, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.{fmt}"
        filepath = os.path.join(out_dir, filename)

        export_cmd = f'Export["{filepath}", {expr}, "{wl_format}"]'
        raw = self._send_command(export_cmd)

        # 等待文件出现（内核IO可能需要时间）
        for _ in range(20):   # 20 * 0.5 = 10秒
            if os.path.exists(filepath):
                return filepath
            time.sleep(0.5)

        # 文件未生成，抛出异常并附上内核输出
        raise RuntimeError(
            f"Export 失败，文件未生成：{filepath}\n"
            f"内核输出：\n{raw}"
        )

    def eval_array(self, expr: str):
        """返回 numpy 数组（需要 numpy）。"""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy required for eval_array()")
        path = self.evaluate_to_file(expr, fmt="json")
        import json
        with open(path) as f:
            data = json.load(f)
        os.unlink(path)
        return np.array(data)

    def eval_image(self, expr: str):
        """返回 PIL Image（需要 Pillow）。"""
        try:
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError("Pillow required for eval_image()")
        path = self.evaluate_to_file(expr, fmt="png")
        img = PILImage.open(path)
        img.load()
        os.unlink(path)
        return img

    # ---------- 批量 evaluate ----------
    def batch_evaluate(self, exprs: list[str]) -> list[str]:
        holders = []
        for expr in exprs:
            h = {"result": None, "error": None, "event": threading.Event()}
            self._req_queue.put((expr, h))
            holders.append(h)
        results = []
        for h in holders:
            h["event"].wait()
            if h["error"]:
                results.append(f"ERROR: {h['error']}")
            else:
                results.append(h["result"])
        return results

    # ---------- 内部串行处理 ----------
    def _serial_loop(self):
        while True:
            item = self._req_queue.get()
            if item is None:
                log.info("Worker 线程收到退出信号")
                break
            expr, holder = item
            try:
                result = self.evaluate(expr)
                holder["result"] = result
                holder["error"] = None
            except Exception as e:
                holder["result"] = None
                holder["error"] = str(e)
                self._restart_kernel()
            finally:
                holder["event"].set()


class WolframPipeline:
    def __init__(self, kernel: WolframKernel):
        self._kernel = kernel
        self._steps = []

    def then(self, expr_template: str, output_key: str = "result") -> "WolframPipeline":
        self._steps.append((expr_template, output_key))
        return self

    def run(self, **inputs) -> dict:
        ctx = dict(inputs)
        for i, (template, out_key) in enumerate(self._steps):
            expr = template.format(**ctx)
            output = self._kernel.evaluate(expr)
            ctx[out_key] = output
            ctx["result"] = output
        return ctx

    def run_array(self, **inputs):
        ctx = dict(inputs)
        for i, (template, out_key) in enumerate(self._steps[:-1]):
            expr = template.format(**ctx)
            output = self._kernel.evaluate(expr)
            ctx[out_key] = output
            ctx["result"] = output
        last_template, last_key = self._steps[-1]
        expr = last_template.format(**ctx)
        output = self._kernel.eval_array(expr)
        ctx[last_key] = output
        ctx["result"] = output
        return ctx
