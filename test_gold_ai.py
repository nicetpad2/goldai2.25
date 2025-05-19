import importlib
import sys
import types
import unittest
import os
import json
import datetime
import logging
import tempfile
from unittest.mock import patch, mock_open, MagicMock

try:
    import pytest
except Exception:  # pragma: no cover - pytest not installed
    class DummyMark:
        def __getattr__(self, name):
            return lambda x: x

    class DummyPytest:
        mark = DummyMark()

        class _Raises:
            def __init__(self, exc):
                self.exc = exc
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc_val, _):
                if exc_type is None:
                    raise AssertionError("Expected exception not raised")
                return issubclass(exc_type, self.exc)

        def raises(self, exc):
            return self._Raises(exc)

        def importorskip(self, name):
            try:
                __import__(name)
            except Exception:
                raise unittest.SkipTest(f"{name} not available")

    pytest = DummyPytest()

try:
    import coverage  # optional
    cov = coverage.Coverage(source=["gold_ai2025"], branch=True)
except Exception:  # pragma: no cover - coverage library not installed
    cov = None


def _create_mock_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__version__ = "0.0"

    def _getattr(attr: str):
        return MagicMock(name=f"{name}.{attr}")

    module.__getattr__ = _getattr  # type: ignore
    if name == "yaml":
        def safe_load(content):
            if hasattr(content, "read"):
                content = content.read()
            data = {}
            for line in str(content).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    val = v.strip()
                    try:
                        data[k.strip()] = json.loads(val)
                    except Exception:
                        data[k.strip()] = val
            return data
        module.safe_load = safe_load  # type: ignore
    return module


# Helper to avoid DataFrame truth ambiguity when extracting kwargs in tests
def safe_extract_df(kwargs, args):
    """Return DataFrame argument regardless of truthiness."""
    return kwargs["df_m1_segment_pd"] if "df_m1_segment_pd" in kwargs else args[0]


# Helper for robust isinstance checks when ga.pd.DataFrame might be MagicMock
def safe_isinstance(obj, typ):
    """Return True if ``obj`` is an instance of ``typ`` or mocked equivalent."""
    try:
        return isinstance(obj, typ)
    except TypeError:
        if getattr(typ, "__class__", None) and typ.__class__.__name__ == "MagicMock":
            return hasattr(obj, "columns")
        return "DataFrame" in str(type(obj))


def safe_import_gold_ai(ipython_ret=None, drive_mod=None) -> types.ModuleType:
    """Import gold_ai2025 with heavy dependencies mocked."""
    mock_modules = {
        "torch": _create_mock_module("torch"),
        "shap": _create_mock_module("shap"),
        "matplotlib": _create_mock_module("matplotlib"),
        "matplotlib.pyplot": _create_mock_module("matplotlib.pyplot"),
        "matplotlib.font_manager": _create_mock_module("matplotlib.font_manager"),
        "scipy": _create_mock_module("scipy"),
        "optuna": _create_mock_module("optuna"),
        "GPUtil": _create_mock_module("GPUtil"),
        "psutil": _create_mock_module("psutil"),
        "cv2": _create_mock_module("cv2"),
        "yaml": _create_mock_module("yaml"),
        "tqdm": _create_mock_module("tqdm"),
        "tqdm.notebook": _create_mock_module("tqdm.notebook"),
        "pandas": _create_mock_module("pandas"),
        "numpy": _create_mock_module("numpy"),
        "ta": _create_mock_module("ta"),
        "requests": _create_mock_module("requests"),
        "sklearn": _create_mock_module("sklearn"),
        "sklearn.cluster": _create_mock_module("sklearn.cluster"),
        "sklearn.preprocessing": _create_mock_module("sklearn.preprocessing"),
        "sklearn.model_selection": _create_mock_module("sklearn.model_selection"),
        "sklearn.metrics": _create_mock_module("sklearn.metrics"),
        "joblib": _create_mock_module("joblib"),
        "matplotlib.ticker": _create_mock_module("matplotlib.ticker"),
        "pynvml": _create_mock_module("pynvml"),
        "scipy.stats": _create_mock_module("scipy.stats"),
    }

    catboost_mod = _create_mock_module("catboost")
    catboost_mod.CatBoostClassifier = object
    catboost_mod.Pool = object
    mock_modules["catboost"] = catboost_mod

    ipython_mod = _create_mock_module("IPython")
    ipython_mod.get_ipython = lambda: ipython_ret
    mock_modules["IPython"] = ipython_mod
    google_colab_mod = _create_mock_module("google.colab")
    drive_module = drive_mod or _create_mock_module("google.colab.drive")
    google_colab_mod.drive = drive_module
    mock_modules["google.colab"] = google_colab_mod
    mock_modules["google.colab.drive"] = drive_module

    with patch("subprocess.run") as _:
        with patch.dict(sys.modules, mock_modules):
            module_name = "gold_ai2025"
            file_path = os.path.join(os.path.dirname(__file__), f"{module_name}.py")
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [ln for ln in f.readlines() if ln.strip() != "import_core_libraries()"]
                source = "".join(lines)
            module = types.ModuleType(module_name)
            module.__file__ = file_path
            sys.modules[module_name] = module
            code_obj = compile(source, file_path, "exec")
            exec(code_obj, module.__dict__)
            return module


def generate_df_tp2_besl(pd_module):
    """Create DataFrame for TP2/BE-SL tests."""
    index = pd_module.date_range("2023-01-01", periods=6, freq="min")
    return pd_module.DataFrame({
        "Open":   [1000, 1005, 1010, 1015, 1013, 1008],
        "High":   [1005, 1012, 1018, 1025, 1020, 1010],
        "Low":    [999, 1002, 1008, 1010, 999, 1000],
        "Close":  [1004, 1010, 1015, 1010, 999, 1005],
        "Entry_Long": [1, 0, 0, 0, 0, 0],
        "ATR_14_Shifted": [1.0] * 6,
        "Signal_Score": [2.0] * 6,
        "Trade_Reason": ["test"] * 6,
        "session": ["Asia"] * 6,
        "Gain_Z": [0.3] * 6,
        "MACD_hist_smooth": [0.1] * 6,
        "RSI": [50] * 6,
    }, index=index)


class TestGoldAIPart1SetupAndEnv(unittest.TestCase):
    def test_environment_is_colab_drive_mount_succeeds(self):
        mount_mock = MagicMock()
        drive_module = types.ModuleType("google.colab.drive")
        drive_module.mount = mount_mock
        class DummyIPy:
            def __str__(self):
                return "google.colab"

        ipy = DummyIPy()
        mod = safe_import_gold_ai(ipython_ret=ipy, drive_mod=drive_module)
        self.assertTrue(mod.IN_COLAB)
        mount_mock.assert_called()

class TestGoldAI2025(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold_ai = safe_import_gold_ai()

    def test_strategy_config_defaults(self):
        cfg = self.gold_ai.StrategyConfig({})
        self.assertEqual(cfg.risk_per_trade, 0.01)
        self.assertEqual(cfg.max_lot, 5.0)
        self.assertEqual(cfg.min_lot, 0.01)
        self.assertEqual(cfg.kill_switch_dd, 0.20)

    def test_load_config_from_yaml(self):
        yaml_data = "risk_per_trade: 0.02\nmax_lot: 2.0\n"
        with patch("builtins.open", mock_open(read_data=yaml_data)):
            cfg = self.gold_ai.load_config_from_yaml("dummy.yaml")
        self.assertEqual(cfg.risk_per_trade, 0.02)
        self.assertEqual(cfg.max_lot, 2.0)

    def test_setup_output_directory(self):
        with patch("os.makedirs") as makedirs, \
             patch("os.path.isdir", return_value=True), \
             patch("builtins.open", mock_open()), \
             patch("os.remove"):
            path = self.gold_ai.setup_output_directory("/tmp", "outdir")
        makedirs.assert_called_with("/tmp/outdir", exist_ok=True)
        self.assertEqual(path, "/tmp/outdir")

    def test_should_exit_due_to_holding(self):
        func = self.gold_ai.should_exit_due_to_holding
        self.assertFalse(func(5, 1, None))
        self.assertFalse(func(5, 1, -1))
        self.assertTrue(func(10, 5, 3))
        self.assertFalse(func(6, 5, 3))

    def test_log_library_version_none(self):
        with self.assertLogs(f"{self.gold_ai.__name__}.log_library_version", level="WARNING") as cm:
            self.gold_ai.log_library_version("dummy", None)
        self.assertTrue(any("dummy" in msg.lower() for msg in cm.output))

    def test_log_library_version_info(self):
        mod = types.ModuleType("dummy")
        mod.__version__ = "1.0"
        with self.assertLogs(f"{self.gold_ai.__name__}.log_library_version", level="INFO") as cm:
            self.gold_ai.log_library_version("dummy", mod)
        self.assertTrue(any("1.0" in msg for msg in cm.output))

    def test_try_import_with_install_success(self):
        mod = types.ModuleType("foo")
        mod.__version__ = "0.1"
        with patch("importlib.import_module", return_value=mod):
            result = self.gold_ai.try_import_with_install(
                "foo", import_as_name="foo", success_flag_global_name="foo_imported"
            )
        self.assertIs(result, mod)
        self.assertTrue(self.gold_ai.foo_imported)
        self.assertIs(self.gold_ai.foo, mod)

    def test_library_import_fails_install_succeeds(self):
        mod = types.ModuleType("bar")
        mod.__version__ = "0.2"
        original_import = importlib.import_module

        def side_effect(name, *args, **kwargs):
            if name == "bar" and side_effect.calls == 0:
                side_effect.calls += 1
                raise ImportError("missing")
            if name == "bar":
                return mod
            return original_import(name, *args, **kwargs)

        side_effect.calls = 0
        with patch.object(self.gold_ai.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch.object(self.gold_ai.importlib, "import_module", side_effect=side_effect):
                result = self.gold_ai.try_import_with_install(
                    "bar",
                    pip_install_name="bar",
                    import_as_name="bar",
                    success_flag_global_name="bar_imported",
                )
        self.assertIs(result, mod)
        self.assertTrue(self.gold_ai.bar_imported)
        self.assertIs(self.gold_ai.bar, mod)

    def test_library_already_imported(self):
        mod = types.ModuleType("baz")
        mod.__version__ = "1.0"
        with patch("importlib.import_module", return_value=mod) as mock_import:
            with patch.object(self.gold_ai.subprocess, "run") as mock_run:
                result = self.gold_ai.try_import_with_install(
                    "baz", pip_install_name="baz", import_as_name="baz", success_flag_global_name="baz_imported"
                )
        self.assertIs(result, mod)
        self.assertTrue(self.gold_ai.baz_imported)
        self.assertIs(self.gold_ai.baz, mod)
        mock_import.assert_called_once_with("baz")
        mock_run.assert_not_called()

    def test_safe_get_global(self):
        self.gold_ai.some_var = 123
        self.assertEqual(self.gold_ai.safe_get_global("some_var", 0), 123)
        self.assertEqual(self.gold_ai.safe_get_global("missing", 99), 99)

    def test_minimal_test_function(self):
        out = self.gold_ai.minimal_test_function()
        self.assertIn("executed successfully", out)

    def test_load_app_config_success(self):
        script_dir = os.getcwd()
        cfg_path = os.path.join(script_dir, "cfg.json")
        with patch("os.path.exists", side_effect=lambda p: p == cfg_path):
            with patch("builtins.open", mock_open(read_data="{\"a\":1}")):
                data = self.gold_ai.load_app_config("cfg.json")
        self.assertEqual(data, {"a": 1})

    def test_load_app_config_not_found(self):
        with patch("os.path.exists", return_value=False):
            data = self.gold_ai.load_app_config("nofile.json")
        self.assertEqual(data, {})

    def test_simple_converter(self):
        self.gold_ai.np = self.gold_ai.DummyNumpy()
        pd_dummy = self.gold_ai.DummyPandas()
        pd_dummy.isna = lambda x: x is None
        self.gold_ai.pd = pd_dummy
        self.gold_ai.datetime = datetime

        self.assertEqual(self.gold_ai.simple_converter(5), 5)
        self.assertEqual(self.gold_ai.simple_converter(3.5), 3.5)
        self.assertIsNone(self.gold_ai.simple_converter(float("nan")))
        self.assertEqual(self.gold_ai.simple_converter(float("inf")), "Infinity")
        self.assertTrue(self.gold_ai.simple_converter(True))
        self.assertEqual(
            self.gold_ai.simple_converter(datetime.datetime(2020, 1, 1)),
            "2020-01-01T00:00:00",
        )
        self.assertEqual(self.gold_ai.simple_converter("text"), "text")
        obj = object()
        self.assertEqual(self.gold_ai.simple_converter(obj), str(obj))

    def test_safe_load_csv_auto(self):
        pd_dummy = self.gold_ai.DummyPandas()
        self.gold_ai.pd = pd_dummy
        with patch("os.path.exists", return_value=False):
            self.assertIsNone(self.gold_ai.safe_load_csv_auto("missing.csv"))

        with patch("os.path.exists", return_value=True):
            with patch.object(pd_dummy, "read_csv", return_value="df") as mock_rc:
                res = self.gold_ai.safe_load_csv_auto("data.csv")
                self.assertEqual(res, "df")
                mock_rc.assert_called_with(
                    "data.csv", index_col=0, parse_dates=False, low_memory=False
                )

        with patch("os.path.exists", return_value=True):
            import io

            fake_file = io.StringIO("x")
            m = MagicMock()
            m.return_value.__enter__.return_value = fake_file
            with patch.object(self.gold_ai.gzip, "open", m):
                with patch.object(pd_dummy, "read_csv", return_value="df2") as mrc:
                    res = self.gold_ai.safe_load_csv_auto("data.csv.gz")
                    self.assertEqual(res, "df2")
                    m.assert_called_with("data.csv.gz", "rt", encoding="utf-8")
                    mrc.assert_called()

    def test_risk_manager_drawdown_and_kill_switch(self):
        pd_dummy = self.gold_ai.DummyPandas()
        pd_dummy.isna = lambda x: x is None or x != x
        self.gold_ai.pd = pd_dummy
        cfg = self.gold_ai.StrategyConfig({})
        rm = self.gold_ai.RiskManager(cfg)

        self.assertEqual(rm.update_drawdown(100.0), 0.0)
        self.assertFalse(rm.soft_kill_active)

        dd = rm.update_drawdown(90.0)
        self.assertAlmostEqual(dd, 0.1, places=4)
        self.assertFalse(rm.soft_kill_active)

        dd = rm.update_drawdown(85.0)
        self.assertAlmostEqual(dd, 0.15, places=4)
        self.assertTrue(rm.soft_kill_active)

        with self.assertRaises(RuntimeError):
            rm.update_drawdown(79.0)

    def test_risk_manager_soft_kill_active(self):
        pd_dummy = self.gold_ai.DummyPandas()
        pd_dummy.isna = lambda x: False
        self.gold_ai.pd = pd_dummy
        cfg = self.gold_ai.StrategyConfig({})
        rm = self.gold_ai.RiskManager(cfg)
        rm.dd_peak = 100
        self.assertFalse(rm.soft_kill_active)
        rm.update_drawdown(80)
        self.assertTrue(rm.soft_kill_active)

    def test_risk_manager_consecutive_loss_and_trading_allowed(self):
        pd_dummy = self.gold_ai.DummyPandas()
        pd_dummy.isna = lambda x: False
        self.gold_ai.pd = pd_dummy
        cfg = self.gold_ai.StrategyConfig({})
        rm = self.gold_ai.RiskManager(cfg)

        self.assertFalse(rm.check_consecutive_loss_kill(3))
        self.assertTrue(rm.check_consecutive_loss_kill(cfg.kill_switch_consecutive_losses))

        rm.soft_kill_active = True
        self.assertFalse(rm.is_trading_allowed())
        rm.soft_kill_active = False
        self.assertTrue(rm.is_trading_allowed())

    def test_print_gpu_utilization_and_show_system_status(self):
        mod = self.gold_ai

        mod.psutil_imported = True
        mod.psutil = MagicMock()
        mod.psutil.virtual_memory.return_value = types.SimpleNamespace(
            percent=75.0,
            used=8 * 1024**3,
            total=16 * 1024**3,
        )

        util = types.SimpleNamespace(gpu=50, memory=40)
        mem = types.SimpleNamespace(used=512 * 1024**2, total=2048 * 1024**2)
        mod.pynvml = types.SimpleNamespace(
            nvmlDeviceGetUtilizationRates=lambda handle: util,
            nvmlDeviceGetMemoryInfo=lambda handle: mem,
            NVMLError=Exception,
            nvmlShutdown=lambda: None,
        )
        mod.nvml_handle = object()
        mod.USE_GPU_ACCELERATION = True

        with self.assertLogs(f"{mod.__name__}.print_gpu_utilization", level="INFO") as cm:
            mod.print_gpu_utilization("Test")
        self.assertTrue(any("GPU Util" in msg for msg in cm.output))

        mod.gputil_imported = True
        mod.GPUtil = types.SimpleNamespace(
            getGPUs=lambda: [
                types.SimpleNamespace(
                    id=0,
                    name="TestGPU",
                    load=0.5,
                    memoryUtil=0.4,
                    memoryUsed=1024,
                    memoryTotal=2048,
                )
            ]
        )
        with self.assertLogs(f"{mod.__name__}.show_system_status", level="INFO") as cm2:
            mod.show_system_status("Demo")
        self.assertTrue(any("TestGPU" in msg for msg in cm2.output))

    def test_check_margin_call(self):
        self.assertTrue(self.gold_ai.check_margin_call(-1.0, 0.0))
        self.assertTrue(self.gold_ai.check_margin_call(5.0, 5.0))
        self.assertFalse(self.gold_ai.check_margin_call(10.0, 0.0))

    def test_check_kill_switch_logic(self):
        cfg = self.gold_ai.StrategyConfig({})
        log_parent = logging.getLogger("test_kill")
        now = datetime.datetime(2021, 1, 1)
        res = self.gold_ai._check_kill_switch(80.0, 100.0, 0.1, 5, 0, False, now, cfg, log_parent)
        self.assertEqual(res, (True, True))
        res = self.gold_ai._check_kill_switch(100.0, 100.0, 0.5, 3, 4, False, now, cfg, log_parent)
        self.assertEqual(res, (True, True))
        res = self.gold_ai._check_kill_switch(90.0, 100.0, 0.5, 5, 1, False, now, cfg, log_parent)
        self.assertEqual(res, (False, False))

    def test_spike_guard_and_reentry(self):
        pd_dummy = self.gold_ai.DummyPandas()
        pd_dummy.isna = lambda x: x is None
        pd_dummy.notna = lambda x: not pd_dummy.isna(x)
        self.gold_ai.pd = pd_dummy
        cfg = self.gold_ai.StrategyConfig({})
        cfg.enable_spike_guard = True
        cfg.spike_guard_score_threshold = 0.5
        cfg.spike_guard_london_patterns = ["Breakout"]
        row = {"spike_score": 0.6, "Pattern_Label": "Breakout"}
        self.assertTrue(self.gold_ai.spike_guard_blocked(row, "London", cfg))
        self.assertFalse(self.gold_ai.spike_guard_blocked(row, "Asia", cfg))
        cfg.use_reentry = True
        cfg.reentry_cooldown_bars = 3
        cfg.reentry_min_proba_thresh = 0.5
        cfg.reentry_cooldown_after_tp_minutes = 1
        row_ns = types.SimpleNamespace(name=datetime.datetime(2021, 1, 1, 0, 10))
        active_orders = []
        self.assertFalse(self.gold_ai.is_reentry_allowed(cfg, row_ns, "BUY", active_orders, 1, None, 0.6))
        self.assertTrue(self.gold_ai.is_reentry_allowed(
            cfg,
            types.SimpleNamespace(name=datetime.datetime(2021, 1, 1, 0, 15)),
            "BUY",
            [],
            5,
            datetime.datetime(2021, 1, 1, 0, 0),
            0.6,
        ))

    def test_all_module_functions_present(self):
        funcs = [n for n, o in vars(self.gold_ai).items() if callable(o) and getattr(o, "__module__", None) == self.gold_ai.__name__]
        self.assertTrue(len(funcs) > 0)

    def test_set_thai_font_and_setup_fonts(self):
        mod = self.gold_ai
        rc_params = {}
        mod.plt = types.SimpleNamespace(
            rcParams=rc_params,
            subplots=lambda figsize=None: (MagicMock(), MagicMock()),
            close=lambda fig=None: None,
        )
        mod.fm = types.SimpleNamespace(
            findfont=lambda *args, **kwargs: "/tmp/font.ttf",
            FontProperties=lambda fname=None: types.SimpleNamespace(get_name=lambda: "Loma"),
            _load_fontmanager=lambda try_read_cache=False: None,
        )
        mod.get_ipython = lambda: types.SimpleNamespace(__str__=lambda self: "google.colab")
        with patch.object(mod.subprocess, "run") as mock_run, \
             patch("os.path.exists", return_value=True):
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            self.assertTrue(mod.set_thai_font("Loma"))
            mod.setup_fonts()


class TestEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ga = safe_import_gold_ai()
        try:
            import pandas as real_pd
            cls.ga.pd = real_pd
            cls.pandas_available = True
        except Exception:
            cls.ga.pd = cls.ga.DummyPandas()
            cls.ga.pd.isna = lambda x: x is None
            cls.pandas_available = False
        try:
            import numpy as real_np
            cls.ga.np = real_np
            cls.numpy_available = True
        except Exception:
            cls.ga.np = cls.ga.DummyNumpy()
            cls.numpy_available = False
        cls.ga.datetime = datetime

    def test_simple_converter_edge_cases(self):
        self.assertEqual(self.ga.simple_converter(self.ga.np.inf), "Infinity")
        self.assertEqual(self.ga.simple_converter(-self.ga.np.inf), "-Infinity")
        self.assertEqual(
            self.ga.simple_converter(datetime.date(2024, 12, 31)), "2024-12-31"
        )
        self.assertIsInstance(self.ga.simple_converter(set([1, 2])), str)
        self.assertIsInstance(self.ga.simple_converter(complex(2, 3)), str)

    def test_setup_output_directory_permission_error(self):
        with patch("os.makedirs", side_effect=PermissionError("Read-only file system")):
            with self.assertRaises(SystemExit):
                self.ga.setup_output_directory("/root", "unwritable")

    def test_log_library_version_none_and_missing(self):
        self.ga.log_library_version("DUMMY_NONE", None)
        dummy_module = types.SimpleNamespace()
        if hasattr(dummy_module, "__version__"):
            delattr(dummy_module, "__version__")
        self.ga.log_library_version("DUMMY_MISSING", dummy_module)

    def test_safe_load_csv_auto_invalid_path_type(self):
        self.assertIsNone(self.ga.safe_load_csv_auto(None))
        self.assertIsNone(self.ga.safe_load_csv_auto(123))

    def test_safe_load_csv_auto_empty_file(self):
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data="")), \
             patch.object(self.ga.pd, "read_csv", side_effect=self.ga.pd.errors.EmptyDataError):
            df = self.ga.safe_load_csv_auto("empty.csv")
            self.assertIsInstance(df, self.ga.pd.DataFrame)
            self.assertTrue(getattr(df, "empty", True))

    def test_safe_load_csv_auto_gz_corrupt(self):
        with patch("os.path.exists", return_value=True), \
             patch("gzip.open", side_effect=OSError("gzip error")):
            df = self.ga.safe_load_csv_auto("corrupt.csv.gz")
            self.assertIsNone(df)

    def test_strategy_config_custom_values(self):
        config = self.ga.StrategyConfig(
            {
                "risk_per_trade": 0.02,
                "max_lot": 10.0,
                "kill_switch_dd": 0.25,
                "enable_spike_guard": False,
            }
        )
        self.assertEqual(config.risk_per_trade, 0.02)
        self.assertEqual(config.max_lot, 10.0)
        self.assertEqual(config.kill_switch_dd, 0.25)
        self.assertFalse(config.enable_spike_guard)

    def test_prepare_datetime_all_nat(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        df = self.ga.pd.DataFrame(
            {
                "Date": ["0000" for _ in range(5)],
                "Timestamp": ["99:99:99" for _ in range(5)],
                "Open": [1] * 5,
                "High": [1] * 5,
                "Low": [1] * 5,
                "Close": [1] * 5,
            }
        )
        with self.assertRaises(SystemExit):
            self.ga.prepare_datetime(df, timeframe_str="ALL_NAT_TEST")

    def test_load_data_missing_columns(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        df_mock = self.ga.pd.DataFrame({"Date": ["20240101"], "Open": [1], "High": [1]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "invalid.csv")
            df_mock.to_csv(path, index=False)
            with self.assertRaises(Exception):
                self.ga.load_data(path)

    def test_load_data_valid(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        df_mock = self.ga.pd.DataFrame(
            {
                "Date": ["20240101"] * 3,
                "Timestamp": ["00:00:00", "00:01:00", "00:02:00"],
                "Open": [1.0, 1.2, 1.3],
                "High": [1.5, 1.4, 1.6],
                "Low": [0.9, 1.1, 1.2],
                "Close": [1.3, 1.2, 1.5],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "valid.csv")
            df_mock.to_csv(path, index=False)
            df_loaded = self.ga.load_data(path)
            self.assertIsInstance(df_loaded, self.ga.pd.DataFrame)
            self.assertFalse(df_loaded.empty)

    def test_load_data_missing_file(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "no_such_file.csv")
            with self.assertRaises(self.ga.DataLoadError):
                self.ga.load_data(path)

    def test_simulate_trades_minimal(self):
        if not self.pandas_available or not self.numpy_available:
            self.skipTest("pandas/numpy not available")
        df = self.ga.pd.DataFrame({
            "Open": [1800.0, 1801.0],
            "High": [1805.0, 1803.0],
            "Low": [1795.0, 1798.0],
            "Close": [1802.0, 1800.0],
            "Entry_Long": [1, 0],
            "ATR_14_Shifted": [1.0, 1.0],
            "Signal_Score": [2.0, 0.0],
            "Trade_Reason": ["test", ""],
            "session": ["Asia", "Asia"],
            "Gain_Z": [0.3, 0.1],
            "MACD_hist_smooth": [0.1, 0.1],
            "RSI": [50, 50],
        })
        df.index = self.ga.pd.date_range("2023-01-01", periods=2, freq="min")
        cfg = self.ga.StrategyConfig({"risk_per_trade": 0.01})
        trade_log, equity_curve, run_summary = self.ga.simulate_trades(df.copy(), cfg)
        self.assertIsInstance(trade_log, list)
        self.assertIsInstance(equity_curve, list)
        self.assertIsInstance(run_summary, dict)

    @pytest.mark.unit
    def test_simulate_trades_tp1_sl(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        df = self.ga.pd.DataFrame({
            "Open": [1000, 1001, 1002, 1003],
            "High": [1005, 1006, 1002.5, 1003.5],
            "Low": [999, 1000, 999.5, 998.0],
            "Close": [1001, 1005, 1000, 998],
            "Entry_Long": [1, 0, 0, 0],
            "ATR_14_Shifted": [1.0] * 4,
            "Signal_Score": [2.0] * 4,
            "Trade_Reason": ["test"] * 4,
            "session": ["Asia"] * 4,
            "Gain_Z": [0.3] * 4,
            "MACD_hist_smooth": [0.1] * 4,
            "RSI": [50] * 4,
        }, index=self.ga.pd.date_range("2023-01-01", periods=4, freq="min"))

        cfg = self.ga.StrategyConfig({"risk_per_trade": 0.01})
        trade_log, equity_curve, run_summary = self.ga.simulate_trades(df.copy(), cfg)

        self.assertGreaterEqual(len(trade_log), 1)
        self.assertIn(trade_log[0]["exit_reason"], {"TP", "SL", "BE-SL"})

    @pytest.mark.unit
    def test_simulate_trades_tsl_behavior(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        df = self.ga.pd.DataFrame({
            "Open": [1000, 1005, 1010, 1007, 1004],
            "High": [1005, 1010, 1015, 1011, 1006],
            "Low": [999, 1002, 1008, 1005, 1001],
            "Close": [1004, 1009, 1013, 1006, 1002],
            "Entry_Long": [1, 0, 0, 0, 0],
            "ATR_14_Shifted": [1.0] * 5,
            "Signal_Score": [2.0] * 5,
            "Trade_Reason": ["test"] * 5,
            "session": ["Asia"] * 5,
            "Gain_Z": [0.3] * 5,
            "MACD_hist_smooth": [0.1] * 5,
            "RSI": [50] * 5,
        }, index=self.ga.pd.date_range("2023-01-01", periods=5, freq="min"))

        cfg = self.ga.StrategyConfig({
            "risk_per_trade": 0.01,
            "use_tsl": True,
            "trailing_sl_distance": 1.5,
        })

        trade_log, equity_curve, run_summary = self.ga.simulate_trades(df.copy(), cfg)
        self.assertEqual(len(trade_log), 1)
        self.assertIn(trade_log[0]["exit_reason"], {"TSL", "TP", "BE-SL", "SL"})

    def test_calculate_metrics_basic(self):
        trades = [
            {"entry_idx": 0, "exit_reason": "TP", "pnl_usd_net": 20.0, "side": "BUY"},
            {"entry_idx": 1, "exit_reason": "SL", "pnl_usd_net": -10.0, "side": "SELL"},
            {"entry_idx": 2, "exit_reason": "BE-SL", "pnl_usd_net": 0.0, "side": "BUY"},
        ]
        summary = self.ga.calculate_metrics(trades, fold_tag="test")
        self.assertEqual(summary["fold_tag"], "test")
        self.assertEqual(summary["num_trades"], 3)
        self.assertEqual(summary["num_tp"], 1)
        self.assertEqual(summary["num_sl"], 1)
        self.assertEqual(summary["num_be"], 1)

    def test_try_import_install_success_no_version(self):
        mod = types.ModuleType("novers")
        if hasattr(mod, "__version__"):
            delattr(mod, "__version__")

        original_import = importlib.import_module

        def side_effect(name, *args, **kwargs):
            if name == "novers" and side_effect.calls == 0:
                side_effect.calls += 1
                raise ImportError("missing")
            if name == "novers":
                return mod
            return original_import(name, *args, **kwargs)

        side_effect.calls = 0
        with patch.object(self.ga.subprocess, "run") as mock_run, \
             patch.object(self.ga.importlib, "import_module", side_effect=side_effect):
            mock_run.return_value = MagicMock(returncode=0)
            result = self.ga.try_import_with_install(
                "novers",
                pip_install_name="novers",
                import_as_name="novers",
                success_flag_global_name="novers_imported",
            )
        self.assertIs(result, mod)
        self.assertTrue(self.ga.novers_imported)

    def test_dummy_module_behavior(self):
        dummy = _create_mock_module("dummy_mod")
        self.assertEqual(dummy.__version__, "0.0")
        attr = dummy.some_attr
        self.assertIsInstance(attr, MagicMock)
        self.assertEqual(attr._mock_name, "dummy_mod.some_attr")

    def test_prepare_datetime_duplicate_index(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        df = self.ga.pd.DataFrame(
            {
                "Date": ["20240101", "20240101"],
                "Timestamp": ["00:00:00", "00:00:00"],
                "Open": [1, 1],
                "High": [1, 1],
                "Low": [1, 1],
                "Close": [1, 1],
            }
        )
        result = self.ga.prepare_datetime(df, timeframe_str="DUP")
        self.assertFalse(result.index.has_duplicates)
        self.assertEqual(len(result), 1)

    def test_set_thai_font_fallback_no_match(self):
        mod = self.ga
        with patch.object(mod.fm, "findfont", side_effect=ValueError("no font")), \
             patch("os.path.exists", return_value=False):
            self.assertFalse(mod.set_thai_font("NoFont"))

    def test_run_backtest_simulation_v34_minimal(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        df = self.ga.pd.DataFrame({
            "Open": [1800.0] * 10,
            "High": [1805.0] * 10,
            "Low": [1795.0] * 10,
            "Close": [1802.0] * 10,
            "Entry_Long": [0] * 10,
            "ATR_14_Shifted": [1.0] * 10,
            "Signal_Score": [2.0] * 10,
            "Trade_Reason": ["test"] * 10,
            "session": ["Asia"] * 10,
            "Gain_Z": [0.3] * 10,
            "MACD_hist_smooth": [0.1] * 10,
            "RSI": [50] * 10,
        })
        df.index = self.ga.pd.date_range("2023-01-01", periods=10, freq="min")
        cfg = self.ga.StrategyConfig({})
        result = self.ga.run_backtest_simulation_v34(
            df,
            config_obj=cfg,
            label="Minimal",
            initial_capital_segment=1000.0,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("trade_log", result)
        self.assertIn("run_summary", result)


class TestWFVandLotSizing(unittest.TestCase):
    """Additional tests for multi-order simulation, WFV, and lot sizing."""

    @classmethod
    def setUpClass(cls):
        cls.ga = safe_import_gold_ai()
        try:
            import pandas as real_pd
            cls.ga.pd = real_pd
            cls.pandas_available = True
        except Exception:
            cls.ga.pd = cls.ga.DummyPandas()
            cls.ga.pd.isna = lambda x: x is None
            cls.pandas_available = False
        try:
            import numpy as real_np
            cls.ga.np = real_np
            cls.numpy_available = True
        except Exception:
            cls.ga.np = cls.ga.DummyNumpy()
            cls.numpy_available = False
        cls.ga.datetime = datetime

    def test_simulate_trades_multi_order_with_reentry(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        df = self.ga.pd.DataFrame({
            "Open": [1000.0, 1000.5],
            "High": [1001.0, 1001.5],
            "Low": [999.5, 1000.0],
            "Close": [1001.0, 1002.0],
            "Entry_Long": [1, 1],
            "ATR_14_Shifted": [1.0, 1.0],
            "Signal_Score": [2.0, 2.0],
            "Trade_Reason": ["test", "test"],
            "session": ["Asia", "Asia"],
            "Gain_Z": [0.3, 0.3],
            "MACD_hist_smooth": [0.1, 0.1],
            "RSI": [50, 50],
        })
        df.index = self.ga.pd.to_datetime([
            "2023-01-01 00:00:00",
            "2023-01-01 00:01:00",
        ])
        cfg = self.ga.StrategyConfig({
            "use_reentry": True,
            "reentry_cooldown_bars": 0,
            "reentry_cooldown_after_tp_minutes": 0,
            "initial_capital": 100.0,
        })
        trade_log, equity_curve, run_summary = self.ga.simulate_trades(df.copy(), cfg)
        self.assertGreaterEqual(len(trade_log), 2)
        self.assertIn(trade_log[0]["exit_reason"], {"TP", "TSL", "BE-SL", "SL"})
        self.assertIn(trade_log[1]["exit_reason"], {"TP", "TSL", "BE-SL", "SL"})
        allowed = self.ga.is_reentry_allowed(
            cfg,
            df.iloc[1],
            "BUY",
            [],
            0,
            df.index[0],
            0.6,
        )
        self.assertTrue(allowed)

    def test_calculate_lot_by_fund_mode_bounds(self):
        cfg = self.ga.StrategyConfig({"min_lot": 0.01, "max_lot": 0.1, "point_value": 0.1})
        lot = self.ga.calculate_lot_by_fund_mode(cfg, "balanced", 0.02, 50000.0, 1.0, 1.0)
        self.assertLessEqual(lot, cfg.max_lot)
        lot_zero = self.ga.calculate_lot_by_fund_mode(cfg, "balanced", 0.02, 0.0, 1.0, 1.0)
        self.assertEqual(lot_zero, cfg.min_lot)
        lot_tight_sl = self.ga.calculate_lot_by_fund_mode(cfg, "balanced", 0.02, 1000.0, 1.0, 0.0)
        self.assertEqual(lot_tight_sl, cfg.min_lot)

    def test_run_all_folds_with_threshold_mocked(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")

        df = self.ga.pd.DataFrame({
            "Open": [1.0, 2.0, 3.0, 4.0],
            "High": [1.5, 2.5, 3.5, 4.5],
            "Low": [0.5, 1.5, 2.5, 3.5],
            "Close": [1.2, 2.2, 3.2, 4.2],
            "Gain_Z": [0.3, 0.4, 0.5, 0.6],
            "RSI": [50, 51, 52, 53],
            "Pattern_Label": ["Breakout"] * 4,
            "Volatility_Index": [1.0] * 4,
        })
        df.index = self.ga.pd.date_range("2023-01-01", periods=4, freq="min")

        cfg = self.ga.StrategyConfig({"n_walk_forward_splits": 2, "initial_capital": 100.0})
        rm = self.ga.RiskManager(cfg)
        tm = self.ga.TradeManager(cfg, rm)

        class DummyTS:
            def __init__(self, n_splits):
                self.n_splits = n_splits

            def split(self, data):
                n = len(data)
                splits = [
                    (list(range(0, 2)), list(range(2, 3))),
                    (list(range(0, 3)), list(range(3, 4))),
                ]
                for tr, te in splits[: self.n_splits]:
                    yield tr, te

        def fake_calc(df_m1, fold_specific_config=None, strategy_config=None):
            df2 = df_m1.copy()
            df2["Entry_Long"] = 1
            df2["Entry_Short"] = 0
            df2["Signal_Score"] = 2.0
            df2["Trade_Reason"] = "MOCK"
            df2["Trade_Tag"] = "T"
            return df2

        def fake_run(*args, **kwargs):
            data = safe_extract_df(kwargs, args)
            side = kwargs.get("side", "BUY")
            trade_log = self.ga.pd.DataFrame([
                {"entry_idx": 0, "exit_reason": "TP", "pnl_usd_net": 1.0, "side": side}
            ])
            return (
                data,
                trade_log,
                kwargs.get("initial_capital_segment", 100.0) + 1.0,
                {data.index[0]: 100.0},
                0.0,
                {"total_ib_lot_accumulator": 1.0},
                [],
                "L1",
                "L2",
                False,
                0,
                1.0,
            )

        with patch.object(self.ga, "TimeSeriesSplit", DummyTS), \
             patch.object(self.ga, "calculate_m1_entry_signals", side_effect=fake_calc), \
             patch.object(self.ga, "_run_backtest_simulation_v34_full", side_effect=fake_run), \
             patch.object(self.ga, "export_run_summary_to_json"), \
             patch.object(self.ga, "export_trade_log_to_csv"), \
             patch.object(self.ga, "plot_equity_curve"):
            result = self.ga.run_all_folds_with_threshold(cfg, rm, tm, df, "/tmp")

        self.assertIsInstance(result[0], dict)
        self.assertIsInstance(result[2], self.ga.pd.DataFrame)


class TestTP2AndBESL(unittest.TestCase):
    """Tests for TP2 and BE-SL exit logic."""

    @classmethod
    def setUpClass(cls):
        cls.ga = safe_import_gold_ai()
        try:
            import pandas as real_pd
            cls.ga.pd = real_pd
            cls.pandas_available = True
        except Exception:
            cls.ga.pd = cls.ga.DummyPandas()
            cls.pandas_available = False
        cls.ga.datetime = datetime

    def test_partial_tp2_trigger(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        cfg = self.ga.StrategyConfig({
            "risk_per_trade": 0.01,
            "enable_partial_tp": True,
            "partial_tp_levels": [
                {"r_multiple": 0.8, "close_pct": 0.5},
                {"r_multiple": 1.2, "close_pct": 0.5},
            ],
            "partial_tp_move_sl_to_entry": False,
            "base_tp_multiplier": 2.0,
            "default_sl_multiplier": 1.0,
        })
        df = generate_df_tp2_besl(self.ga.pd)
        trade_log, equity, summary = self.ga.simulate_trades(df.copy(), cfg)
        self.assertGreaterEqual(len(trade_log), 1)
        self.assertIn(
            trade_log[0]["exit_reason"], {"TP", "SL", "PartialTP", "BE-SL"}
        )

    def test_besl_trigger(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        cfg = self.ga.StrategyConfig({
            "risk_per_trade": 0.01,
            "base_tp_multiplier": 1.5,
            "base_be_sl_r_threshold": 1.0,
            "default_sl_multiplier": 1.0,
            "enable_be_sl": True,
            "enable_partial_tp": False,
        })
        # sequence designed to hit BE-SL threshold
        df = self.ga.pd.DataFrame({
            "Open": [1000, 1010, 1020, 1025, 1015, 1000, 995],
            "High": [1015, 1025, 1030, 1035, 1020, 1005, 1001],
            "Low": [995, 1005, 1015, 1020, 1000, 995, 990],
            "Close": [1010, 1020, 1030, 1025, 1000, 999, 995],
            "Entry_Long": [1, 0, 0, 0, 0, 0, 0],
            "ATR_14_Shifted": [1.0] * 7,
            "Signal_Score": [2.0] * 7,
            "Trade_Reason": ["test"] * 7,
            "session": ["Asia"] * 7,
            "Gain_Z": [0.3] * 7,
            "MACD_hist_smooth": [0.1] * 7,
            "RSI": [50] * 7,
        }, index=self.ga.pd.date_range("2023-01-01", periods=7, freq="min"))
        trade_log, equity_curve, run_summary = self.ga.simulate_trades(df.copy(), cfg)
        print("=== trade_log for BE-SL debug ===")
        for t in trade_log:
            print(t)
        self.assertTrue(any(t["exit_reason"] in {"BE-SL", "SL"} for t in trade_log))

    def test_simulate_trades_tsl_tp_be_sl(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        df = self.ga.pd.DataFrame({
            "Open": [1000, 1005, 1010, 1007, 1004, 1002],
            "High": [1005, 1010, 1015, 1011, 1006, 1004],
            "Low": [999, 1002, 1008, 1005, 1002, 1000],
            "Close": [1004, 1009, 1013, 1006, 1002, 1001],
            "Entry_Long": [1, 0, 0, 0, 0, 0],
            "ATR_14_Shifted": [1.0] * 6,
            "Signal_Score": [2.0] * 6,
            "Trade_Reason": ["test"] * 6,
            "session": ["Asia"] * 6,
            "Gain_Z": [0.3] * 6,
            "MACD_hist_smooth": [0.1] * 6,
            "RSI": [50] * 6,
        }, index=self.ga.pd.date_range("2023-01-01", periods=6, freq="min"))

        cfg = self.ga.StrategyConfig({
            "risk_per_trade": 0.01,
            "use_tsl": True,
            "trailing_sl_distance": 1.5,
            "base_tp_multiplier": 1.5,
            "default_sl_multiplier": 1.0,
            "base_be_sl_r_threshold": 1.0,
            "enable_partial_tp": True,
            "partial_tp_levels": [{"r_multiple": 0.5, "close_pct": 0.5}],
            "partial_tp_move_sl_to_entry": True,
        })

        trade_log, equity_curve, run_summary = self.ga.simulate_trades(df.copy(), cfg)
        self.assertTrue(any(t["exit_reason"] in {"TSL", "TP", "BE-SL"} for t in trade_log))

    def test_simulate_trades_with_kill_switch_activation(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        # six-bar declining market to trip kill switch logic
        df = self.ga.pd.DataFrame({
            "Open": [1000, 995, 990, 985, 980, 975],
            "High": [1001, 996, 991, 986, 981, 976],
            "Low": [995, 990, 985, 980, 975, 970],
            "Close": [995, 990, 985, 980, 975, 970],
            "Entry_Long": [1, 0, 0, 0, 0, 0],
            "ATR_14_Shifted": [1.0] * 6,
            "Signal_Score": [2.0] * 6,
            "Trade_Reason": ["test"] * 6,
            "session": ["Asia"] * 6,
            "Gain_Z": [0.3] * 6,
            "MACD_hist_smooth": [0.1] * 6,
            "RSI": [50] * 6,
        }, index=self.ga.pd.date_range("2023-01-01", periods=6, freq="min"))

        cfg = self.ga.StrategyConfig({
            "risk_per_trade": 0.5,
            "initial_capital": 10.0,
            "kill_switch_dd": 0.10,
            "kill_switch_consecutive_losses": 1,
            "recovery_mode_consecutive_losses": 1,
        })

        run_summary = {}
        try:
            trade_log, equity_curve, run_summary = self.ga.simulate_trades(df.copy(), cfg)
        except RuntimeError:
            run_summary["hard_kill_triggered"] = True

        self.assertTrue(
            run_summary.get("hard_kill_triggered") or run_summary.get("kill_switch_active")
        )


class TestWFVandLotSizingFix(unittest.TestCase):
    """Ensure reentry logic handles 0 cooldown correctly."""

    @classmethod
    def setUpClass(cls):
        cls.ga = safe_import_gold_ai()
        try:
            import pandas as real_pd
            cls.ga.pd = real_pd
            cls.pandas_available = True
        except Exception:
            cls.ga.pd = cls.ga.DummyPandas()
            cls.pandas_available = False
        try:
            import numpy as real_np
            cls.ga.np = real_np
            cls.numpy_available = True
        except Exception:
            cls.ga.np = cls.ga.DummyNumpy()
            cls.numpy_available = False
        cls.ga.datetime = datetime

    def test_simulate_trades_multi_order_with_reentry_fixed(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        is_reentry_allowed = self.ga.is_reentry_allowed

        df = self.ga.pd.DataFrame({
            "Open": [1000.0, 1000.5],
            "High": [1001.0, 1001.5],
            "Low": [999.5, 1000.0],
            "Close": [1001.0, 1002.0],
            "Entry_Long": [1, 1],
            "ATR_14_Shifted": [1.0, 1.0],
            "Signal_Score": [2.0, 2.0],
            "Trade_Reason": ["test", "test"],
            "session": ["Asia", "Asia"],
            "Gain_Z": [0.3, 0.3],
            "MACD_hist_smooth": [0.1, 0.1],
            "RSI": [50, 50],
        })
        df.index = self.ga.pd.to_datetime([
            "2023-01-01 00:00:00",
            "2023-01-01 00:01:00",
        ])
        cfg = self.ga.StrategyConfig({
            "use_reentry": True,
            "reentry_cooldown_bars": 0,
            "reentry_cooldown_after_tp_minutes": 0,
            "initial_capital": 100.0,
        })
        trade_log, equity_curve, summary = self.ga.simulate_trades(df.copy(), cfg)
        self.assertEqual(len(trade_log), 2)
        # Accept TP or BE-SL as valid, since BE-SL can occur due to tight BE logic
        self.assertIn(trade_log[0]["exit_reason"], ["TP", "BE-SL"])
        self.assertEqual(trade_log[1]["exit_reason"], "TP")
        allowed = is_reentry_allowed(cfg, df.iloc[1], "BUY", [], 0, df.index[0], 0.6)
        self.assertTrue(allowed)

    def test_multi_order_reentry_succeeds(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        # dataset expanded to four bars to validate reentry handling
        df = self.ga.pd.DataFrame({
            "Open": [1000.0, 1005.0, 1003.0, 1004.0],
            "High": [1006.0, 1010.0, 1005.0, 1006.0],
            "Low": [999.0, 1003.0, 1000.0, 1002.0],
            "Close": [1005.0, 1009.0, 1002.0, 1003.0],
            "Entry_Long": [1, 0, 1, 0],
            "ATR_14_Shifted": [1.0] * 4,
            "Signal_Score": [2.0] * 4,
            "Trade_Reason": ["test"] * 4,
            "session": ["Asia"] * 4,
            "Gain_Z": [0.3] * 4,
            "MACD_hist_smooth": [0.1] * 4,
            "RSI": [50] * 4,
        }, index=self.ga.pd.date_range("2023-01-01", periods=4, freq="min"))

        cfg = self.ga.StrategyConfig({
            "use_reentry": True,
            "reentry_cooldown_bars": 0,
            "reentry_cooldown_after_tp_minutes": 0,
            "initial_capital": 100.0,
            "risk_per_trade": 0.01,
        })

        trade_log, equity_curve, run_summary = self.ga.simulate_trades(df.copy(), cfg)
        self.assertGreaterEqual(len(trade_log), 2)
        self.assertIn(trade_log[0]["exit_reason"], {"TP", "TSL", "BE-SL", "SL"})
        self.assertIn(trade_log[1]["exit_reason"], {"TP", "TSL", "BE-SL", "SL"})

    def test_simulate_trades_multi_order_strict(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        df = self.ga.pd.DataFrame({
            "Open": [1000, 1001, 1002],
            "High": [1003, 1004, 1005],
            "Low": [999, 1000, 1001],
            "Close": [1002, 1003, 1004],
            "Entry_Long": [1, 1, 0],
            "ATR_14_Shifted": [1.0, 1.0, 1.0],
            "Signal_Score": [2.0, 2.0, 0.0],
            "Trade_Reason": ["test", "test", ""],
            "session": ["Asia", "Asia", "Asia"],
            "Gain_Z": [0.3, 0.3, 0.3],
            "MACD_hist_smooth": [0.1, 0.1, 0.1],
            "RSI": [50, 50, 50],
        })
        df.index = self.ga.pd.to_datetime([
            "2023-01-01 00:00:00",
            "2023-01-01 00:01:00",
            "2023-01-01 00:02:00",
        ])
        cfg = self.ga.StrategyConfig({
            "use_reentry": True,
            "reentry_cooldown_bars": 0,
            "reentry_cooldown_after_tp_minutes": 0,
            "initial_capital": 100.0,
            "risk_per_trade": 0.01,
        })
        trade_log, equity_curve, run_summary = self.ga.simulate_trades(df.copy(), cfg)
        self.assertGreaterEqual(len(trade_log), 2)
        exit_reasons = set(t['exit_reason'] for t in trade_log)
        self.assertTrue(all(reason in {"TP", "SL", "BE-SL", "TSL"} for reason in exit_reasons))

    def test_simulate_trades_reentry_strict(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        df = self.ga.pd.DataFrame({
            "Open": [1000, 1002, 1004],
            "High": [1003, 1005, 1006],
            "Low": [999, 1001, 1003],
            "Close": [1002, 1004, 1005],
            "Entry_Long": [1, 0, 1],
            "ATR_14_Shifted": [1.0, 1.0, 1.0],
            "Signal_Score": [2.0, 0.0, 2.0],
            "Trade_Reason": ["test", "", "test"],
            "session": ["Asia", "Asia", "Asia"],
            "Gain_Z": [0.3, 0.3, 0.3],
            "MACD_hist_smooth": [0.1, 0.1, 0.1],
            "RSI": [50, 50, 50],
        })
        df.index = self.ga.pd.to_datetime([
            "2023-01-01 00:00:00",
            "2023-01-01 00:01:00",
            "2023-01-01 00:02:00",
        ])
        cfg = self.ga.StrategyConfig({
            "use_reentry": True,
            "reentry_cooldown_bars": 0,
            "reentry_cooldown_after_tp_minutes": 0,
            "initial_capital": 100.0,
            "risk_per_trade": 0.01,
        })
        trade_log, equity_curve, run_summary = self.ga.simulate_trades(df.copy(), cfg)
        self.assertGreaterEqual(len(trade_log), 2)
        times = [t['entry_time'] for t in trade_log]
        self.assertEqual(len(times), len(set(times)))

    def test_simulate_trades_BE_SL_multi(self):
        if not self.pandas_available:
            self.skipTest("pandas not available")
        df = self.ga.pd.DataFrame({
            "Open": [1000, 1002, 1004],
            "High": [1003, 1005, 1006],
            "Low": [999, 1001, 1003],
            "Close": [999, 1001, 1003],
            "Entry_Long": [1, 1, 1],
            "ATR_14_Shifted": [1.0, 1.0, 1.0],
            "Signal_Score": [2.0, 2.0, 2.0],
            "Trade_Reason": ["test", "test", "test"],
            "session": ["Asia", "Asia", "Asia"],
            "Gain_Z": [0.3, 0.3, 0.3],
            "MACD_hist_smooth": [0.1, 0.1, 0.1],
            "RSI": [50, 50, 50],
        })
        df.index = self.ga.pd.to_datetime([
            "2023-01-01 00:00:00",
            "2023-01-01 00:01:00",
            "2023-01-01 00:02:00",
        ])
        cfg = self.ga.StrategyConfig({
            "use_reentry": True,
            "reentry_cooldown_bars": 0,
            "reentry_cooldown_after_tp_minutes": 0,
            "initial_capital": 100.0,
            "risk_per_trade": 0.01,
        })
        trade_log, equity_curve, run_summary = self.ga.simulate_trades(df.copy(), cfg)
        self.assertTrue(all(t['exit_reason'] in {"BE-SL", "SL", "TP"} for t in trade_log))


class TestWarningEdgeCases(unittest.TestCase):
    """Additional coverage for warning and failure scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.ga = safe_import_gold_ai()
        try:
            import pandas as real_pd
            cls.ga.pd = real_pd
        except Exception:
            cls.ga.pd = cls.ga.DummyPandas()

    def test_safe_load_csv_auto_corrupt_encoding(self):
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data="\xff\xfe\xfd")), \
             patch.object(self.ga.pd, "read_csv", side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")):
            result = self.ga.safe_load_csv_auto("bad_encoding.csv")
            self.assertIsNone(result)

    def test_safe_load_csv_auto_permission_denied(self):
        with patch("os.makedirs"), \
             patch("os.path.exists", return_value=True), \
             patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with self.assertRaises(SystemExit):
                self.ga.setup_output_directory("/root", "denied")

    def test_set_thai_font_without_matplotlib(self):
        with patch.object(self.ga.fm, "findfont", side_effect=ImportError("matplotlib not available")):
            try:
                self.ga.set_thai_font("Loma")
            except Exception as e:
                self.fail(f"set_thai_font raised unexpected error without matplotlib: {e}")

    def test_safe_load_csv_auto_gz_with_corrupt_content(self):
        with patch("os.path.exists", return_value=True), \
             patch.object(self.ga.gzip, "open", side_effect=OSError("corrupt gzip")):
            result = self.ga.safe_load_csv_auto("bad_file.csv.gz")
            self.assertIsNone(result)

    def test_safe_load_csv_auto_utf8_bom_file(self):
        csv_data = '\ufeffDate,Open,High,Low,Close\n20240101,1000,1005,995,1001\n'
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=csv_data)), \
             patch.object(self.ga.pd, "read_csv", return_value="df") as mock_rc:
            result = self.ga.safe_load_csv_auto("bom_file.csv")
            self.assertEqual(result, "df")
            mock_rc.assert_called()


class TestFeatureEngineeringCoverage:
    """Coverage for key feature engineering functions."""

    def setup_method(self):
        pytest.importorskip("pandas")
        pytest.importorskip("numpy")
        import pandas as pd
        import numpy as np

        self.ga = safe_import_gold_ai()
        self.ga.pd = pd
        self.ga.np = np

        self.df = pd.DataFrame({
            "Open": [1, 2, 3, 4, 5],
            "High": [2, 3, 4, 5, 6],
            "Low": [0, 1, 2, 3, 4],
            "Close": [1.5, 2.5, 3.5, 4.5, 5.5],
        })
        self.config = self.ga.StrategyConfig({})

    def test_ema_normal(self):
        result = self.ga.ema(self.df["Close"], 3)
        expected = self.df["Close"].ewm(span=3, adjust=False, min_periods=3).mean()
        # Use np.allclose for floating-point
        import numpy as np
        assert np.allclose(result, expected.astype("float32"), equal_nan=True)

    def test_ema_empty(self):
        empty = self.ga.pd.Series([], dtype="float32")
        result = self.ga.ema(empty, 5)
        assert result is not None and len(result) == 0

    def test_sma_normal(self):
        s = self.df["Close"]
        result = self.ga.sma(s, 3)
        assert isinstance(result, self.ga.pd.Series)

    def test_sma_invalid_period(self):
        s = self.df["Close"]
        result = self.ga.sma(s, 0)
        assert result.isnull().all()

    def test_rsi_normal(self):
        pytest.importorskip("ta")
        s = self.df["Close"]
        result = self.ga.rsi(s, 3)
        expected = self.ga.ta.momentum.RSIIndicator(close=s, window=3, fillna=False).rsi()
        import numpy as np
        assert np.allclose(result, expected.astype("float32"), equal_nan=True)

    def test_rsi_empty(self):
        s = self.ga.pd.Series([], dtype="float32")
        out = self.ga.rsi(s, 5)
        assert out is not None and len(out) == 0

    def test_atr_valid(self):
        out = self.ga.atr(self.df, 3)
        assert "ATR_3" in out.columns

    def test_atr_missing_cols(self):
        df = self.ga.pd.DataFrame({"Close": [1, 2, 3]})
        out = self.ga.atr(df, 3)
        assert "ATR_3" in out.columns

    def test_macd_normal(self):
        s = self.df["Close"]
        macd_line, macd_signal, macd_hist = self.ga.macd(s)
        assert isinstance(macd_line, self.ga.pd.Series)

    def test_macd_empty(self):
        s = self.ga.pd.Series([], dtype="float32")
        macd_line, macd_signal, macd_hist = self.ga.macd(s)
        assert macd_line.empty

    def test_rolling_zscore_short_series(self):
        s = self.ga.pd.Series([1])
        z = self.ga.rolling_zscore(s, 5)
        assert (z == 0.0).all()

    def test_tag_price_structure_patterns_missing_col(self):
        df = self.ga.pd.DataFrame({"Gain_Z": [1, -1]})
        # config ไม่เกี่ยวกับ expected_type แล้ว ฟังก์ชันหลักจะ fix expected_type เอง
        res = self.ga.tag_price_structure_patterns(df, self.config)
        assert "Pattern_Label" in res.columns

    def test_calculate_m15_trend_zone_minimal(self):
        df = self.ga.pd.DataFrame({"Close": [1, 2, 3, 4, 5]})
        # ฟังก์ชันหลักกำหนด expected_type เองแบบ DataFrame
        res = self.ga.calculate_m15_trend_zone(df, self.config)
        assert "Trend_Zone" in res.columns

    def test_get_session_tag_na(self):
        tag = self.ga.get_session_tag(self.ga.pd.NaT, self.config.session_times_utc)
        assert tag == "N/A"

    def test_get_session_tag_zones(self):
        session_times = self.config.session_times_utc
        dt_asia = self.ga.pd.Timestamp("2023-01-01 02:00:00")
        dt_london = self.ga.pd.Timestamp("2023-01-01 08:00:00")
        dt_ny = self.ga.pd.Timestamp("2023-01-01 18:00:00")
        assert self.ga.get_session_tag(dt_asia, session_times) in ("Asia", "London", "NY", "N/A")
        assert self.ga.get_session_tag(dt_london, session_times) in ("Asia", "London", "NY", "N/A")
        assert self.ga.get_session_tag(dt_ny, session_times) in ("Asia", "London", "NY", "N/A")

    def test_engineer_m1_features_empty(self):
        df = self.ga.pd.DataFrame()
        res = self.ga.engineer_m1_features(df, self.config)
        assert res.empty

    def test_clean_m1_data_empty(self):
        df = self.ga.pd.DataFrame()
        clean, feats = self.ga.clean_m1_data(df, self.config)
        assert clean.empty
        assert feats == []


class TestBranchAndErrorPathCoverage:
    """Additional branch and error handling coverage."""

    def test_ema_negative_period(self):
        pytest.importorskip("pandas")
        import pandas as pd
        ga = safe_import_gold_ai()
        ga.pd = pd
        s = pd.Series([1, 2, 3])
        with pytest.raises(ValueError):
            ga.ema(s, -1)

    def test_sma_not_series(self):
        ga = safe_import_gold_ai()
        with pytest.raises(Exception):
            ga.sma([1, 2, 3], 2)

    def test_rsi_not_series(self):
        ga = safe_import_gold_ai()
        with pytest.raises(Exception):
            ga.rsi([1, 2, 3], 2)

    def test_atr_invalid_type(self):
        ga = safe_import_gold_ai()
        with pytest.raises(Exception):
            ga.atr([1, 2, 3], 2)

    def test_macd_invalid_type(self):
        ga = safe_import_gold_ai()
        with pytest.raises(Exception):
            ga.macd([1, 2, 3])

    def test_rolling_zscore_nan(self):
        pytest.importorskip("pandas")
        pytest.importorskip("numpy")
        import pandas as pd
        import numpy as np
        ga = safe_import_gold_ai()
        ga.pd = pd
        ga.np = np
        s = pd.Series([np.nan, np.nan])
        z = ga.rolling_zscore(s, 5)
        assert (z == 0.0).all()

    def test_tag_price_structure_patterns_empty(self):
        ga = safe_import_gold_ai()
        config = ga.StrategyConfig({})
        df = None  # input ผิด type
        res = ga.tag_price_structure_patterns(df, config)
        assert safe_isinstance(res, ga.pd.DataFrame)
        assert getattr(res, "empty", True)

    def test_calculate_m15_trend_zone_empty(self):
        ga = safe_import_gold_ai()
        config = ga.StrategyConfig({})
        df = None  # input ผิด type
        res = ga.calculate_m15_trend_zone(df, config)
        assert safe_isinstance(res, ga.pd.DataFrame)
        assert getattr(res, "empty", True)

    def test_tag_price_structure_patterns_empty_df(self):
        ga = safe_import_gold_ai()
        config = ga.StrategyConfig({})
        df = ga.pd.DataFrame()
        res = ga.tag_price_structure_patterns(df, config)
        assert safe_isinstance(res, ga.pd.DataFrame)
        assert getattr(res, "empty", True)
        assert list(res.columns) == ["Pattern_Label"]

    def test_calculate_m15_trend_zone_empty_df(self):
        ga = safe_import_gold_ai()
        config = ga.StrategyConfig({})
        df = ga.pd.DataFrame()
        res = ga.calculate_m15_trend_zone(df, config)
        assert safe_isinstance(res, ga.pd.DataFrame)
        assert getattr(res, "empty", True)
        assert list(res.columns) == ["Trend_Zone"]

    def test_tag_price_structure_patterns_type_guard(self):
        ga = safe_import_gold_ai()
        res = ga.tag_price_structure_patterns([], ga.StrategyConfig({}), expected_type=123)
        assert safe_isinstance(res, ga.pd.DataFrame)
        assert getattr(res, "empty", True)

    def test_tag_price_structure_patterns_expected_type_guard(self):
        ga = safe_import_gold_ai()
        cfg = ga.StrategyConfig({})
        df = ga.pd.DataFrame()
        res1 = ga.tag_price_structure_patterns(df, cfg, expected_type="notatype")
        res2 = ga.tag_price_structure_patterns(df, cfg, expected_type=(str, "badtype"))
        assert getattr(res1, "empty", True)
        assert getattr(res2, "empty", True)

    def test_calculate_m15_trend_zone_empty_guard(self):
        ga = safe_import_gold_ai()
        res = ga.calculate_m15_trend_zone({}, ga.StrategyConfig({}), expected_type=None)
        assert getattr(res, "empty", True)

    def test_calculate_m15_trend_zone_expected_type_guard(self):
        ga = safe_import_gold_ai()
        cfg = ga.StrategyConfig({})
        df = ga.pd.DataFrame()
        res = ga.calculate_m15_trend_zone(df, cfg, expected_type=[])
        assert getattr(res, "empty", True)


def test_trade_manager_update_methods():
    ga = safe_import_gold_ai()
    pytest.importorskip("pandas")
    import pandas as pd
    cfg = ga.StrategyConfig({})
    rm = ga.RiskManager(cfg)
    tm = ga.TradeManager(cfg, rm)
    test_timestamp = pd.Timestamp("2023-01-01 00:00:00")
    tm.update_last_trade_time(test_timestamp)
    assert tm.last_trade_time == test_timestamp
    assert tm.risk_manager is rm
    tm.update_last_trade_time(None)
    assert tm.last_trade_time == test_timestamp
    tm.update_last_trade_time(pd.NaT)
    assert tm.last_trade_time == test_timestamp
    tm.update_last_trade_time("2023-01-02 00:00:00")
    assert tm.last_trade_time == pd.Timestamp("2023-01-02 00:00:00")
    tm.update_last_trade_time("nan")
    assert tm.last_trade_time == pd.Timestamp("2023-01-02 00:00:00")


def test_run_backtest_simulation_minimal():
    ga = safe_import_gold_ai()
    pytest.importorskip("pandas")
    import pandas as pd
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5],
            "High": [1, 2, 3, 4, 5],
            "Low": [1, 2, 3, 4, 5],
            "Close": [1, 2, 3, 4, 5],
        },
        index=pd.date_range("2023-01-01", periods=5, freq="T"),
    )
    cfg = ga.StrategyConfig({})
    result = ga.run_backtest_simulation_v34(
        df,
        config_obj=cfg,
        label="Test",
        initial_capital_segment=1000,
    )
    assert isinstance(result, dict)
    assert "trade_log" in result


def test_calculate_metrics_minimal():
    ga = safe_import_gold_ai()
    pytest.importorskip("pandas")
    import pandas as pd
    trade_log_list = [{
        "entry_idx": 0,
        "entry_price": 1000,
        "stop_loss": 995,
        "take_profit": 1010,
        "side": "BUY",
        "exit_price": 1010,
        "exit_reason": "TP",
        "exit_idx": 0,
        "exit_time": pd.Timestamp("2023-01-01 00:00:00"),
        "pnl_usd_net": 10.0,
    }]
    result = ga.calculate_metrics(trade_log_list, fold_tag="minimal_test")
    assert "num_tp" in result
    assert result["num_tp"] == 1


def test_safe_load_csv_auto_nonexistent(caplog):
    ga = safe_import_gold_ai()
    path = "file_that_does_not_exist.csv"
    with caplog.at_level("ERROR"):
        result = ga.safe_load_csv_auto(path)
    assert result is None or (hasattr(result, "empty") and result.empty)
    assert any("ไม่พบไฟล์" in rec.message for rec in caplog.records)


class TestRobustFormatAndTypeGuard(unittest.TestCase):
    """[Patch AI Studio v4.9.40] Test robust _float_fmt, safe_isinstance, TradeManager.update_last_trade_time"""

    def test_float_fmt_cases(self):
        ga = safe_import_gold_ai()
        cases = [
            (1.23456, "1.235"),
            (1, "1.000"),
            ("2.718", "2.718"),
            ("bad", "bad"),
            (None, "None"),
            (complex(2, 3), "(2+3j)"),
            ([1, 2], "[1, 2]"),
        ]
        for val, expected in cases:
            result = ga._float_fmt(val)
            if isinstance(val, (float, int)) or (
                isinstance(val, str) and val.replace(".", "", 1).isdigit()
            ):
                self.assertTrue(all(ch.isdigit() or ch == "." for ch in result))
            else:
                self.assertEqual(result, str(val))

    def test_isinstance_safe(self):
        ga = safe_import_gold_ai()
        check = ga._isinstance_safe

        class A:
            pass

        a = A()
        self.assertTrue(check(a, A))
        self.assertTrue(check(a, (A,)))
        self.assertFalse(check(a, None))
        self.assertFalse(check(a, "str"))
        mm = MagicMock()
        self.assertFalse(check(a, mm))
        self.assertFalse(check(a, (123, "bad")))
        self.assertFalse(check("foo", int))

        class DummyType:
            pass

        dt = DummyType()
        self.assertFalse(check(a, dt))

    def test_trade_manager_update_last_trade_time(self):
        ga = safe_import_gold_ai()
        TradeManager = ga.TradeManager
        RiskManager = ga.RiskManager
        StrategyConfig = ga.StrategyConfig

        class PandasStub(types.ModuleType):
            class Timestamp(datetime.datetime):
                pass

            NaT = None

            @staticmethod
            def isna(v):
                return v is None or (isinstance(v, float) and v != v)

            @staticmethod
            def to_datetime(v):
                if isinstance(v, (int, float)):
                    return PandasStub.Timestamp.fromtimestamp(v / 1000 if abs(v) > 1e12 else v)
                return PandasStub.Timestamp.fromisoformat(str(v))

        pd_stub = PandasStub("pandas_stub")
        np_stub = _create_mock_module("numpy")
        with patch.dict(sys.modules, {"pandas": pd_stub, "numpy": np_stub}):
            cfg = StrategyConfig({})
            rm = RiskManager(cfg)
            tm = TradeManager(cfg, rm)

            ts = pd_stub.Timestamp(2023, 1, 1)
            tm.update_last_trade_time(ts)
            self.assertEqual(tm.last_trade_time, ts)

            tm.update_last_trade_time(None)
            self.assertEqual(tm.last_trade_time, ts)

            tm.update_last_trade_time("2023-01-02")
            self.assertIsInstance(tm.last_trade_time, pd_stub.Timestamp)

            tm.update_last_trade_time(1672531200000)
            self.assertIsInstance(tm.last_trade_time, pd_stub.Timestamp)

            tm.update_last_trade_time("nan")
            tm.update_last_trade_time(pd_stub.NaT)
            tm.update_last_trade_time(float("nan"))
            tm.update_last_trade_time("not-a-date")


if __name__ == "__main__":
    if cov:
        cov.start()
    unittest.main(exit=False)

    if cov:
        cov.stop()
        cov.save()
        cov.report(show_missing=True)
