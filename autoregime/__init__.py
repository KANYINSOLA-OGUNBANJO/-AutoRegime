# autoregime/__init__.py
from __future__ import annotations
from typing import Any, Optional, Dict

__version__ = "0.1.0"
__description__ = "Automatic market regime detection with HMM and BOCPD"
__author__ = "Kanyinsola Ogunbanjo"

# ---------------------------------------------------------
# Ensure NumPy compatibility across environments
# ---------------------------------------------------------
try:
    # Prefer the in-package helper if present
    from .ar_np_compat import ensure_numpy_compat  # type: ignore
except Exception:
    # Fallback to repo-root module (dev installs)
    from ar_np_compat import ensure_numpy_compat  # type: ignore
ensure_numpy_compat()

# ---------------------------------------------------------
# Engine entry points (HMM is always available)
# ---------------------------------------------------------
from .engines.hmm_sticky import (  # noqa: E402
    stable_regime_analysis as _hmm_analyze,
    stable_report as _hmm_report,
)

# BOCPD is optional; import lazily and fail gracefully
try:
    from .engines.bocpd import bocpd_regime_analysis as _bocpd_analyze  # type: ignore[attr-defined]
    _bocpd_import_error: Exception | None = None
except Exception as _e:  # pragma: no cover
    _bocpd_analyze = None  # type: ignore
    _bocpd_import_error = _e


def has_bocpd() -> bool:
    """
    Return True if the BOCPD engine imported successfully, False otherwise.
    Useful for apps to toggle availability without raising.
    """
    return _bocpd_analyze is not None


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
    """
    Unified entrypoint for regime analysis.

    Parameters
    ----------
    assets : str | pandas.Series | pandas.DataFrame
        - Ticker string (e.g., "SPY"), or
        - Series of adjusted prices (Date index), or
        - DataFrame of prices (first column used).
    method : {"hmm","bocpd"}
        Detection engine to use. Default "hmm".
    start_date, end_date : str | None
        Date bounds if `assets` is a ticker.
    return_result : bool
        When True, return dict with "report", "regime_timeline", "meta".
    verbose : bool
        Pass-through verbosity flag to engines.
    **kwargs
        Engine-specific settings:
          - HMM: min_segment_days, sticky, n_components (optional)
          - BOCPD: min_segment_days, hazard

    Returns
    -------
    dict
      Keys: "report": str, "regime_timeline": list[dict], "meta": dict
    """
    m = (method or "hmm").lower()
    if m == "hmm":
        return _hmm_analyze(
            assets,
            start_date=start_date,
            end_date=end_date,
            return_result=return_result,
            verbose=verbose,
            **kwargs,
        )
    if m == "bocpd":
        if _bocpd_analyze is None:
            raise ImportError(
                "BOCPD engine is unavailable. Ensure `autoregime/engines/bocpd.py` exists "
                "and that your install includes it (e.g., `pip install -e .`).\n"
                f"Root cause during import: {_bocpd_import_error!r}"
            ) from _bocpd_import_error
        return _bocpd_analyze(
            assets,
            start_date=start_date,
            end_date=end_date,
            return_result=return_result,
            verbose=verbose,
            **kwargs,
        )
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
    """
    Convenience wrapper: returns a formatted professional report string.

    For HMM, this routes to the engine's native reporter.
    For BOCPD, it runs analysis and extracts the 'report' field.
    """
    m = (method or "hmm").lower()
    if m == "hmm":
        return _hmm_report(
            assets,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,
            **kwargs,
        )
    # For non-HMM engines we compute and return the text
    res = stable_regime_analysis(
        assets,
        method=method,
        start_date=start_date,
        end_date=end_date,
        return_result=True,
        verbose=verbose,
        **kwargs,
    )
    return str(res.get("report", ""))


# Friendly aliases
analyze = stable_regime_analysis
report = stable_report

__all__ = [
    "__version__",
    "__description__",
    "__author__",
    "has_bocpd",
    "stable_regime_analysis",
    "stable_report",
    "analyze",
    "report",
]