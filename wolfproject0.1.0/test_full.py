"""
test_full.py
------------
WolframKernel 全功能集成测试
在没有真实 Wolfram 内核的环境里用 mock 验证所有逻辑
"""

import sys
import os
import threading
import unittest
from unittest.mock import patch, MagicMock

# 把桥梁所在目录加入路径
sys.path.insert(0, os.path.dirname(__file__))
import wolfram_bridge
from wolfram_bridge import WolframKernel


# ───────────────────────────────────────────────────────────────
#  Mock：模拟 pexpect.spawn 行为
# ───────────────────────────────────────────────────────────────
class MockChild:
    """模拟 Wolfram 内核的 pexpect 子进程"""

    def __init__(self):
        self._alive = True
        self._out_counter = 1
        self._buf = ""
        self._last_expr = ""

    def isalive(self):
        return self._alive

    def sendline(self, line):
        self._last_expr = line

    def expect(self, pattern, timeout=30):
        import re
        # 匹配启动提示符
        if hasattr(pattern, 'pattern') and 'In' in pattern.pattern:
            self.before = ""
            return 0
        if isinstance(pattern, str) and "__WBRIDGE_" in pattern:
            # 哨兵匹配：构造假输出
            self.before = f"\nOut[{self._out_counter}]= 4\n"
            self._out_counter += 1
            return 0
        if isinstance(pattern, str) and 'In' in pattern:
            self.before = ""
            return 0
        # 编译正则
        if isinstance(pattern, type(re.compile(""))):
            self.before = ""
            return 0
        self.before = ""
        return 0

    def terminate(self, force=False):
        self._alive = False


# ───────────────────────────────────────────────────────────────
#  测试套件
# ───────────────────────────────────────────────────────────────
class TestWolframBridge(unittest.TestCase):

    def setUp(self):
        """每个测试前重置单例"""
        with WolframKernel._instance_lock:
            WolframKernel._instance = None

    def tearDown(self):
        """每个测试后清理单例"""
        with WolframKernel._instance_lock:
            if WolframKernel._instance is not None:
                inst = WolframKernel._instance
                WolframKernel._instance = None
                # 停止 worker
                inst._req_queue.put(None)

    def _make_kernel(self):
        mock_child = MockChild()
        with patch("pexpect.spawn", return_value=mock_child):
            k = WolframKernel()
        return k, mock_child

    # ── 节点2：单例 ────────────────────────────────────────────
    def test_singleton(self):
        print("\n[Test] 单例模式")
        mock_child = MockChild()
        with patch("pexpect.spawn", return_value=mock_child):
            k1 = WolframKernel()
            k2 = WolframKernel()
        self.assertIs(k1, k2, "应返回同一个实例")
        print("  ✅ k1 is k2")

    # ── 节点3：哨兵输出解析 ────────────────────────────────────
    def test_parse_output_with_out(self):
        print("\n[Test] 输出解析 - Out[n]= 格式")
        raw = "\n  2+2\n\nOut[1]= 4\n\n"
        result = WolframKernel._parse_output(raw)
        self.assertEqual(result, "4")
        print(f"  ✅ 解析结果：{result!r}")

    def test_parse_output_print_only(self):
        print("\n[Test] 输出解析 - 纯 Print 输出")
        raw = "hello world\n"
        result = WolframKernel._parse_output(raw)
        self.assertEqual(result, "hello world")
        print(f"  ✅ 解析结果：{result!r}")

    def test_parse_output_multiline(self):
        print("\n[Test] 输出解析 - 多行 Out")
        raw = "Out[1]= {1, 2,\n   3, 4}\n"
        result = WolframKernel._parse_output(raw)
        self.assertIn("1, 2", result)
        print(f"  ✅ 解析结果：{result!r}")

    # ── 节点4：串行队列 ────────────────────────────────────────
    def test_serial_execution(self):
        print("\n[Test] 串行执行顺序")
        k, _ = self._make_kernel()
        order = []
        lock  = threading.Lock()

        def call(expr, idx):
            k.evaluate(expr)
            with lock:
                order.append(idx)

        threads = [threading.Thread(target=call, args=(f"{i}+0", i))
                   for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(order), 5, "所有5个请求都应完成")
        print(f"  ✅ 完成顺序：{order}")

    # ── 节点5：资源管理 ────────────────────────────────────────
    def test_context_manager(self):
        print("\n[Test] with 语句资源管理")
        mock_child = MockChild()
        with patch("pexpect.spawn", return_value=mock_child):
            with WolframKernel() as k:
                self.assertIsNotNone(k)
        # with 块退出后单例应被清除
        self.assertIsNone(WolframKernel._instance)
        print("  ✅ 退出 with 后单例已清理")

    # ── 节点6：错误恢复 ────────────────────────────────────────
    def test_error_recovery(self):
        print("\n[Test] 错误恢复 - 内核崩溃后自动重启")
        call_count = {"n": 0}
        mock_child = MockChild()

        def make_child(*a, **kw):
            call_count["n"] += 1
            return MockChild()

        with patch("pexpect.spawn", side_effect=make_child):
            k = WolframKernel()
            # 模拟内核崩溃
            k._child._alive = False
            k._restart_kernel()

        self.assertGreaterEqual(call_count["n"], 2,
                                "崩溃后应重新 spawn 内核")
        print(f"  ✅ spawn 调用次数：{call_count['n']}（包含初始启动）")

    # ── 节点7：批量计算 ────────────────────────────────────────
    def test_batch_evaluate(self):
        print("\n[Test] 批量计算 - 保序返回")
        k, mock = self._make_kernel()

        # 让 mock 依次返回不同结果
        counter = {"n": 0}
        fake_results = ["1", "2", "3", "4", "5"]

        original_saw = WolframKernel._send_and_wait

        def fake_saw(self_inner, expr):
            r = fake_results[counter["n"] % len(fake_results)]
            counter["n"] += 1
            return r

        with patch.object(WolframKernel, "_send_and_wait", fake_saw):
            results = k.batch_evaluate(["1", "2", "3", "4", "5"])

        self.assertEqual(len(results), 5)
        self.assertEqual(results, fake_results)
        print(f"  ✅ 批量结果（保序）：{results}")

    # ── 综合：多线程并发安全 ────────────────────────────────────
    def test_thread_safety(self):
        print("\n[Test] 多线程并发安全")
        k, _ = self._make_kernel()
        results = [None] * 10
        errors  = []

        def worker(idx):
            try:
                results[idx] = k.evaluate(f"{idx}*{idx}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"不应有错误：{errors}")
        self.assertTrue(all(r is not None for r in results),
                        "所有请求都应返回结果")
        print(f"  ✅ 10 个并发请求全部完成，无错误")


# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestWolframBridge)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
