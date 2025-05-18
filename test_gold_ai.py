import importlib
import sys
import types
import unittest
import os
import json
import datetime
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


def safe_import_gold_ai() -> types.ModuleType:
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
    ipython_mod.get_ipython = lambda: None
    mock_modules["IPython"] = ipython_mod
    mock_modules["google.colab"] = _create_mock_module("google.colab")
    mock_modules["google.colab.drive"] = _create_mock_module("google.colab.drive")

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

    def test_try_import_with_install_after_install(self):
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


if __name__ == "__main__":
    unittest.main()
