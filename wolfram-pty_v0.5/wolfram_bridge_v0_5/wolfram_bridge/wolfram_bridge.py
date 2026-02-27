"""
wolfram_bridge.py
-----------------
稳定的 Wolfram-Python 桥梁
零科学库依赖，所有数据通过文件交换。
  1. 劝告锁竞态   → 单例 + 进程级互斥（节点2）
  2. 串行执行     → 请求队列 + 严格顺序（节点4）
  3. 输入堆积     → 哨兵确认 + 两段等待（节点3）
  4. 资源管理     → context manager（节点5）
  5. 错误恢复     → 自动重启内核（节点6）
  6. 批量计算     → batch_evaluate（节点7）
"""

"""
wolfram_bridge.py
-----------------
稳定的 Wolfram-Python 桥梁，支持文件输出模式与缓存。
"""

import threading
import queue
import uuid
import time
import re
import os
import logging
import tempfile
import hashlib
import shutil

import pexpect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("wolfram_bridge")

# ==================== 可配置常量 ====================
WOLFRAM_EXEC = os.environ.get(
    "WOLFRAM_EXEC",
    "/root/wolfram-extract/opt/Wolfram/WolframEngine/14.1/Executables/math"
)
WOLFRAM_ARGS = [
    "-pwfile", os.path.expanduser("~/.Wolfram/Licensing/mathpass"),
    "-noinit"
]
PROMPT_RE   = re.compile(r"In\[\d+\]:=")
STARTUP_TIMEOUT  = 60
EVAL_TIMEOUT     = 120
SENTINEL_TIMEOUT = 10

# 缓存目录（可自定义）
CACHE_DIR = os.environ.get("WOLFRAM_CACHE_DIR", "/sdcard/wolfram_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _compute_hash(expr: str) -> str:
    """计算表达式的 SHA256 哈希，作为缓存键"""
    return hashlib.sha256(expr.encode('utf-8')).hexdigest()

# ==================== 核心内核类 ====================
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

        self._child: pexpect.spawn | None = None
        self._kernel_lock = threading.Lock()
        self._req_queue: queue.Queue = queue.Queue()
        self._worker_thread = threading.Thread(
            target=self._serial_loop,
            name="wolfram-worker",
            daemon=True
        )

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
            timeout=STARTUP_TIMEOUT
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

    def _send_and_wait(self, expr: str) -> str:
        sentinel = f"__WBRIDGE_{uuid.uuid4().hex}__"
        wrapped = f'({expr}); Print["{sentinel}"]'

        with self._kernel_lock:
            child = self._child

        if child is None or not child.isalive():
            raise RuntimeError("内核未运行")

        child.sendline(wrapped)

        try:
            child.expect(re.escape(sentinel),
                         timeout=EVAL_TIMEOUT + SENTINEL_TIMEOUT)
        except pexpect.TIMEOUT:
            raise TimeoutError(f"表达式求值超时（>{EVAL_TIMEOUT}s）：{expr[:80]}")
        except pexpect.EOF:
            raise RuntimeError("内核在求值中途退出")

        raw = child.before

        try:
            child.expect(PROMPT_RE, timeout=SENTINEL_TIMEOUT)
        except Exception:
            pass

        return self._parse_output(raw)

    @staticmethod
    def _parse_output(raw: str) -> str:
        lines = raw.strip().splitlines()
        for line in reversed(lines):
            m = re.match(r"Out\[\d+\]=\s*(.*)", line)
            if m:
                return m.group(1).strip()
        return "\n".join(
            l for l in lines
            if l.strip() and not PROMPT_RE.search(l)
        ).strip()

    def _serial_loop(self):
        while True:
            item = self._req_queue.get()
            if item is None:
                log.info("Worker 线程收到退出信号")
                break

            expr, holder = item
            try:
                result = self._send_and_wait(expr)
                holder["result"] = result
                holder["error"]  = None
            except Exception as e:
                log.error(f"求值失败：{e}，尝试恢复...")
                holder["result"] = None
                holder["error"]  = str(e)
                try:
                    self._restart_kernel()
                except Exception as re_err:
                    log.error(f"重启失败：{re_err}")
            finally:
                holder["event"].set()

    def evaluate(self, expr: str) -> str:
        holder = {"result": None, "error": None, "event": threading.Event()}
        self._req_queue.put((expr, holder))
        holder["event"].wait()
        if holder["error"]:
            raise RuntimeError(holder["error"])
        return holder["result"]

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

    # ========== 文件输出模式（带缓存） ==========
    def evaluate_to_file(self, expr: str, fmt: str = "json", out_dir: str = "/sdcard/wolfram_out", no_cache: bool = False) -> str:
        """
        将表达式结果导出到文件，返回文件路径。
        支持缓存：相同表达式（字符串完全相同）的结果会被复用，除非 no_cache=True。
        格式 fmt 支持常见扩展名：json, txt, csv, png, pdf, svg, ...
        """
        # 格式映射
        fmt_map = {
            "json": "JSON",
            "txt": "Text",
            "text": "Text",
            "csv": "CSV",
            "tsv": "TSV",
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
            "wdx": "WDX",
            "tex": "TeX",
            "table": "Table",
            "list": "List",
            "mx": "MX",
        }
        wl_format = fmt_map.get(fmt.lower())
        if wl_format is None:
            raise ValueError(f"不支持的格式 '{fmt}'。支持格式：{list(fmt_map.keys())}")

        # 检查缓存
        cache_file = None
        if not no_cache:
            expr_hash = _compute_hash(expr)
            cache_file = os.path.join(CACHE_DIR, f"{expr_hash}.{fmt}")
            if os.path.exists(cache_file):
                log.debug(f"缓存命中：{expr_hash}.{fmt}")
                return cache_file

        # 生成输出文件路径（临时目录或指定 out_dir）
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.{fmt}"
        filepath = os.path.join(out_dir, filename)

        # 执行导出
        export_cmd = f'Export["{filepath}", {expr}, "{wl_format}"]'
        self._send_and_wait(export_cmd)

        # 等待文件出现（最多10秒）
        for _ in range(20):
            if os.path.exists(filepath):
                # 如果启用缓存，将文件复制到缓存目录
                if cache_file is not None:
                    shutil.copy2(filepath, cache_file)
                    # 可选：删除原始文件以节省空间，这里保留 out_dir 中的副本
                return filepath
            time.sleep(0.5)

        raise RuntimeError(f"Export 失败，文件未生成：{filepath}")

    def clear_cache(self, older_than_days: int = None):
        """清理缓存文件。
        若 older_than_days 为 None，则清空所有缓存；
        否则删除早于指定天数的文件。
        """
        import time
        now = time.time()
        count = 0
        for f in os.listdir(CACHE_DIR):
            filepath = os.path.join(CACHE_DIR, f)
            if os.path.isfile(filepath):
                if older_than_days is None:
                    os.remove(filepath)
                    count += 1
                else:
                    age_days = (now - os.path.getmtime(filepath)) / 86400
                    if age_days > older_than_days:
                        os.remove(filepath)
                        count += 1
        log.info(f"已清理 {count} 个缓存文件")
        return count


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