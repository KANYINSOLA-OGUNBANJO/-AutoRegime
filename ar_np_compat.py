# ar_np_compat.py
"""
Make older libs that still use removed NumPy names (np.int, np.float, etc.)
work under NumPy 2.x by defining safe aliases.
Call ensure_numpy_compat() before importing those libs.
"""
def ensure_numpy_compat() -> None:
    import numpy as np  # local import to avoid early import side effects
    # map removed aliases to builtins if missing
    fallback = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }
    for name, typ in fallback.items():
        if not hasattr(np, name):
            setattr(np, name, typ)
    # some libs reference np.bool_ vs np.bool; keep both
    if not hasattr(np, "bool_"):
        np.bool_ = bool  # type: ignore[attr-defined]