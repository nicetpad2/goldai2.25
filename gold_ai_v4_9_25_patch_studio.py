"""
Helper utilities for the studio test environment.
Provides `extend_safe_import_for_studio` which augments
the mock module dictionary used by `test_gold_ai.py`.
"""

from unittest.mock import MagicMock
import logging


def extend_safe_import_for_studio(safe_mock_modules_dict):
    """Add extra mocked modules required by the tests."""
    logger = logging.getLogger("extend_safe_import_for_studio")
    additional = {
        "matplotlib": MagicMock(name="SafeMock_matplotlib"),
        "matplotlib.pyplot": MagicMock(name="SafeMock_matplotlib_pyplot"),
        "scipy": MagicMock(name="SafeMock_scipy"),
    }
    for mod_name, mock_obj in additional.items():
        mock_obj.__version__ = "0.0"
        safe_mock_modules_dict.setdefault(mod_name, mock_obj)
    logger.info("Extended SafeImport mocks for studio environment.")
    return safe_mock_modules_dict
