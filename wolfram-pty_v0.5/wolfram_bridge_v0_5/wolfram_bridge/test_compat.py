import sys, os, json, tempfile, unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

ROOT = str(Path(__file__).parent.parent)
if ROOT not in sys.path: sys.path.insert(0, ROOT)

for k in list(sys.modules): 
    if "wolfram_bridge" in k: del sys.modules[k]

MAPPINGS_DIR = str(Path(__file__).parent / "compat" / "mappings")
os.environ["WOLFRAM_MAPPINGS_DIR"] = MAPPINGS_DIR


class MockKernel:
    def evaluate(self, expr):
        import re
        m = re.match(r'Export\["(.+?)",\s*.+?,\s*"(\w+)"\]', expr)
        if m:
            path, fmt = m.group(1), m.group(2)
            if fmt == "JSON":
                if   "Fourier"     in expr: data = [[1.0,0.0],[-1.0,0.0],[1.0,0.0],[-1.0,0.0]]
                elif "LinearSolve" in expr: data = [-1.0, 3.0]
                elif "Total"       in expr: data = 10
                elif "Mean"        in expr: data = 2.5
                elif "Sort"        in expr: data = [1,2,3,4,5]
                else:                       data = [1.0,2.0,3.0]
                with open(path,"w") as f: json.dump(data, f)
            return f'"{path}"'
        if "Det"  in expr: return "1."
        if "Norm" in expr: return "3.74166"
        return "ok"


class TestMetadata(unittest.TestCase):
    def setUp(self):
        from wolfram_bridge.compat._core.metadata import MetadataRepository
        MetadataRepository._instance = None
        self.repo = MetadataRepository(MAPPINGS_DIR)

    def test_load(self):
        self.assertGreater(len(self.repo.all_rules), 0)
        print(f"\n[Test] YAML 加载 ✅  {len(self.repo.all_rules)} 条规则")

    def test_exact(self):
        rule = self.repo.get_rule("numpy.fft.fft")
        self.assertIsNotNone(rule)
        self.assertEqual(rule["wolfram_function"], "Fourier")
        print(f"\n[Test] 精确查找 ✅  numpy.fft.fft → Fourier")

    def test_tag(self):
        results = self.repo.search_rules("fft")
        self.assertGreater(len(results), 0)
        print(f"\n[Test] 标签搜索 fft ✅  {[r['python_path'] for r in results]}")

    def test_prefix(self):
        results = self.repo.search_rules("numpy.linalg")
        self.assertGreater(len(results), 0)
        print(f"\n[Test] 前缀搜索 numpy.linalg ✅  {len(results)} 条")

    def test_missing(self):
        self.assertIsNone(self.repo.get_rule("numpy.abc.xyz"))
        print("\n[Test] 不存在路径 ✅  正确返回 None")


class TestConverters(unittest.TestCase):
    def test_to_wl_list(self):
        from wolfram_bridge.compat._core.converters import to_wl_list
        self.assertEqual(to_wl_list([1,2,3]), "{1, 2, 3}")
        print("\n[Test] to_wl_list ✅")

    def test_to_wl_matrix(self):
        from wolfram_bridge.compat._core.converters import to_wl_matrix
        r = to_wl_matrix([[1,2],[3,4]])
        self.assertIn("{1, 2}", r)
        print(f"\n[Test] to_wl_matrix ✅  {r}")

    def test_from_wl_json_file(self):
        from wolfram_bridge.compat._core.converters import from_wl_json
        with tempfile.NamedTemporaryFile(mode="w",suffix=".json",delete=False) as f:
            json.dump([1,2,3], f); fpath = f.name
        try:
            self.assertEqual(from_wl_json(fpath), [1,2,3])
            print("\n[Test] from_wl_json 文件模式 ✅")
        finally:
            os.unlink(fpath)

    def test_from_wl_json_str(self):
        from wolfram_bridge.compat._core.converters import from_wl_json
        self.assertEqual(from_wl_json('[1,2,3]'), [1,2,3])
        print("\n[Test] from_wl_json 字符串回退 ✅")

    def test_scalar(self):
        from wolfram_bridge.compat._core.converters import from_wl_scalar
        self.assertEqual(from_wl_scalar("42"), 42)
        print("\n[Test] from_wl_scalar ✅")


class TestResolver(unittest.TestCase):
    def setUp(self):
        from wolfram_bridge.compat._core.metadata import MetadataRepository
        from wolfram_bridge.compat._core.resolver import ResolutionEngine
        MetadataRepository._instance = None
        ResolutionEngine._instance   = None
        self.res = ResolutionEngine(MetadataRepository(MAPPINGS_DIR))

    def test_resolve(self):
        rule = self.res.resolve("numpy.fft.fft")
        self.assertIsNotNone(rule)
        print(f"\n[Test] Resolver 精确匹配 ✅  → {rule['wolfram_function']}")

    def test_file_mode(self):
        rule = self.res.resolve("numpy.fft.fft")
        expr, tmp = self.res.build_wl_expr(rule, ([1,2,3,4],), {})
        self.assertIsNotNone(tmp)
        self.assertIn("Export", expr)
        self.assertIn("Fourier", expr)
        if os.path.exists(tmp): os.unlink(tmp)
        print(f"\n[Test] 文件模式表达式 ✅  {expr[:60]}...")

    def test_scalar_mode(self):
        rule = self.res.resolve("numpy.linalg.det")
        expr, tmp = self.res.build_wl_expr(rule, ([[1,2],[3,4]],), {})
        self.assertIsNone(tmp)
        self.assertIn("Det", expr)
        print(f"\n[Test] 标量模式表达式 ✅  {expr}")

    def test_missing(self):
        self.assertIsNone(self.res.resolve("numpy.xyz.unknown"))
        print("\n[Test] Resolver 未知路径 ✅  返回 None")


class TestProxy(unittest.TestCase):
    def setUp(self):
        for key in list(sys.modules.keys()):
            if "wolfram_bridge" in key:
                del sys.modules[key]
        from wolfram_bridge.compat._core.metadata import MetadataRepository
        from wolfram_bridge.compat._core.resolver import ResolutionEngine
        MetadataRepository._instance = None
        ResolutionEngine._instance   = None

    def _inject(self):
        from wolfram_bridge.compat._state import _state
        _state["kernel"] = MockKernel()

    def test_fft(self):
        import wolfram_bridge.compat.numpy as _np
        self._inject()
        result = _np.fft.fft([1,0,1,0])
        self.assertIsNotNone(result)
        print(f"\n[Test] np.fft.fft 端到端 ✅  type={type(result).__name__}")

    def test_unknown_raises(self):
        import wolfram_bridge.compat.numpy as _np
        self._inject()
        with self.assertRaises(AttributeError):
            _np.abc.xyz([1,2,3])
        print("\n[Test] 未知函数 AttributeError ✅")

    def test_repr(self):
        import wolfram_bridge.compat.numpy as _np
        self.assertIn("numpy", repr(_np))
        print(f"\n[Test] repr ✅  {repr(_np)}")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestMetadata, TestConverters, TestResolver, TestProxy]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
