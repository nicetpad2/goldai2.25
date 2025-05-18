import importlib
import sys
import types
import unittest
import os
import json
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


if __name__ == "__main__":
    unittest.main()
