# autoregime/ar_np_compat.py
"""
Tiny NumPy compatibility shim so older libs (hmmlearn, sklearn versions, etc.)
that reference deprecated NumPy aliases keep working on modern NumPy (>=1.24/2.x).

Call ensure_numpy_compat() once at import time.
"""
from __future__ import annotations

def ensure_numpy_compat() -> None:
    import numpy as np  # noqa: F401

    # Provide deprecated aliases if missing
    for old, new in [
        ("float", float),
        ("int", int),
        ("bool", bool),
        ("object", object),
        ("complex", complex),
        ("long", int),  # Py2 legacy
    ]:
        if not hasattr(np, old):
            try:
                setattr(np, old, new)  # type: ignore[attr-defined]
            except Exception:
                pass

    # You can also relax FP errors if needed (commented by default)
    # np.seterr(all="ignore")