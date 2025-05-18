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
            sys.modules[module_name] = module
            exec(source, module.__dict__)
            return module


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


if __name__ == "__main__":
    try:
        import coverage
        cov = coverage.Coverage()
        cov.start()
    except Exception:
        cov = None

    unittest.main(exit=False)

    if cov:
        cov.stop()
        cov.save()
        cov.report()
