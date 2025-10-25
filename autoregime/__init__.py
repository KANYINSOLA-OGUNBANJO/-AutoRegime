# autoregime/__init__.py
from __future__ import annotations
from typing import Any, Optional, Dict, Iterable
import pandas as _pd

__version__ = "0.1.0"
__description__ = "Automatic market regime detection with HMM and BOCPD"
__author__ = "Kanyinsola Ogunbanjo"

# ensure NumPy compat first
try:
    # if you later move ar_np_compat inside the package, the first import will work
    from .ar_np_compat import ensure_numpy_compat  # type: ignore
except Exception:
    # fall back to repo-root module
    from ar_np_compat import ensure_numpy_compat  # type: ignore
ensure_numpy_compat()

from .engines.hmm_sticky import (
    stable_regime_analysis as _hmm_analyze,
    stable_report as _hmm_report,
)

try:
    from .engines.bocpd import bocpd_regime_analysis as _bocpd_analyze  # type: ignore[attr-defined]
    _bocpd_import_error = None
except Exception as _e:
    _bocpd_analyze = None  # type: ignore
    _bocpd_import_error = _e

def stable_regime_analysis(
    assets: Any,
    *,
    method: str = "hmm",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    return_result: bool = True,
    verbose: bool = False,
    **kwargs,
) -> Dict:
    m = (method or "hmm").lower()
    if m == "hmm":
        return _hmm_analyze(assets, start_date=start_date, end_date=end_date, return_result=return_result, verbose=verbose, **kwargs)
    if m == "bocpd":
        if _bocpd_analyze is None:
            raise ImportError(
                "BOCPD engine failed to import. Ensure autoregime/engines/bocpd.py exists "
                "and reinstall with `pip install -e .`.\n"
                f"Root cause: {_bocpd_import_error!r}"
            ) from _bocpd_import_error
        return _bocpd_analyze(assets, start_date=start_date, end_date=end_date, return_result=return_result, verbose=verbose, **kwargs)
    raise ValueError(f"Unknown method '{method}'. Use 'hmm' or 'bocpd'.")

def stable_report(
    assets: Any,
    *,
    method: str = "hmm",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    verbose: bool = False,
    **kwargs,
) -> str:
    m = (method or "hmm").lower()
    if m == "hmm":
        return _hmm_report(assets, start_date=start_date, end_date=end_date, verbose=verbose, **kwargs)
    res = stable_regime_analysis(assets, method=method, start_date=start_date, end_date=end_date, return_result=True, verbose=verbose, **kwargs)
    return res.get("report", "")