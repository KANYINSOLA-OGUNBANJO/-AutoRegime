# autoregime/ar_np_compat.py
"""
Small NumPy compatibility shim so older deps (e.g., hmmlearn/scikit) don't break on NumPy 2.x.
Safe no-ops if attributes already exist.
"""
from __future__ import annotations

def ensure_numpy_compat() -> None:
    try:
        import numpy as np  # noqa
    except Exception:
        return

    # Add deprecated aliases if missing (harmless if already present)
    fallback = {
        "float": float,
        "int": int,
        "bool": bool,
        "object": object,
        "complex": complex,
        "long": int,  # old alias
    }
    for name, pytype in fallback.items():
        if not hasattr(np, name):
            try:
                setattr(np, name, pytype)
            except Exception:
                pass