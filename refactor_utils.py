"""Utility helpers extracted from gold_ai2025.

[Patch AI Studio v4.9.104] Move `_safe_numeric` from main module for reusability.
"""

from typing import Any, Optional
import logging


def _safe_numeric(val: Any, default: float = 0.0, *, nan_as: Optional[float] = None, log_ctx: str = "") -> float:
    """Convert ``val`` to float safely.

    Handles strings, ``None``, pandas NA/NaT, and other objects. Returns
    ``default`` or ``nan_as`` if conversion fails.
    """
    try:
        import pandas as pd  # Local import for mocks
        import numpy as np

        if val is None or (hasattr(pd, "isna") and pd.isna(val)):
            return nan_as if nan_as is not None else default
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)

        conv = pd.to_numeric(val, errors="coerce")
        if pd.isna(conv):
            return nan_as if nan_as is not None else default
        return float(conv)
    except Exception as exc:  # pragma: no cover - unexpected types
        logging.error(
            f"[Patch AI Studio v4.9.43+] _safe_numeric: Failed in {log_ctx}: {exc}"
        )
        return nan_as if nan_as is not None else default

__all__ = ["_safe_numeric"]
