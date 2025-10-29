# autoregime/__init__.py
from __future__ import annotations
from typing import Any, Optional

__all__ = ["stable_regime_analysis", "stable_report", "has_bocpd", "analyze", "report", "__version__"]

__version__ = "0.1.0"

# -----------------------------------------------------------------------------
# Try to import HMM engine (primary)
# -----------------------------------------------------------------------------
_hmm_analyze = None
_hmm_report_fn = None
_hmm_import_error: Exception | None = None
try:
    from .engines.hmm_sticky import (  # type: ignore
        stable_regime_analysis as _hmm_analyze,
    )
    try:
        # optional (older engine may not export this)
        from .engines.hmm_sticky import stable_report as _hmm_report_fn  # type: ignore
    except Exception:
        _hmm_report_fn = None
except Exception as e:
    _hmm_import_error = e

def _ensure_hmm_available() -> None:
    if _hmm_analyze is None:
        raise ImportError(
            "HMM engine failed to import. Check 'autoregime/engines/hmm_sticky.py' and its "
            "dependencies (hmmlearn, scikit-learn, numpy)."
        ) from _hmm_import_error

# -----------------------------------------------------------------------------
# Try to import BOCPD engine (optional)
# -----------------------------------------------------------------------------
_boc_analyze = None
_boc_import_error: Exception | None = None
try:
    from .engines.bocpd import bocpd_regime_analysis as _boc_analyze  # type: ignore
except Exception as e:
    _boc_import_error = e

def has_bocpd() -> bool:
    return _boc_analyze is not None

def _ensure_bocpd_available() -> None:
    if _boc_analyze is None:
        raise ImportError(
            "BOCPD engine is unavailable. Ensure 'autoregime/engines/bocpd.py' exists and imports cleanly."
        ) from _boc_import_error

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def stable_regime_analysis(
    assets: Any,
    *,
    method: str = "hmm",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    return_result: bool = True,
    verbose: bool = False,
    **kwargs: Any,
):
    """
    Unified entrypoint. Returns the selected engine's result dict (when return_result=True)
    or a formatted report string (when return_result=False).
    """
    m = (method or "hmm").lower()

    if m == "hmm":
        _ensure_hmm_available()
        out = _hmm_analyze(  # type: ignore[misc]
            assets,
            start_date=start_date,
            end_date=end_date,
            return_result=True,  # always get structured; adapt below
            verbose=verbose,
            **kwargs,
        )
        if return_result:
            return out
        return (out.get("report") if isinstance(out, dict) else out)  # type: ignore[return-value]

    if m == "bocpd":
        _ensure_bocpd_available()
        out = _boc_analyze(  # type: ignore[misc]
            assets,
            start_date=start_date,
            end_date=end_date,
            return_result=True,
            verbose=verbose,
            **kwargs,
        )
        if return_result:
            return out
        return (out.get("report") if isinstance(out, dict) else out)  # type: ignore[return-value]

    raise ValueError(f"Unknown method: {method!r}. Expected 'hmm' or 'bocpd'.")


def stable_report(
    assets: Any,
    *,
    method: str = "hmm",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> str:
    """
    Text-only convenience. Uses an engine-provided helper when available (HMM),
    otherwise runs analysis and returns the 'report' field.
    """
    m = (method or "hmm").lower()

    if m == "hmm":
        _ensure_hmm_available()
        if callable(_hmm_report_fn):
            return _hmm_report_fn(  # type: ignore[misc]
                assets, start_date=start_date, end_date=end_date, verbose=verbose, **kwargs
            )
        out = stable_regime_analysis(
            assets,
            method="hmm",
            start_date=start_date,
            end_date=end_date,
            return_result=True,
            verbose=verbose,
            **kwargs,
        )
        return str(out.get("report", "")) if isinstance(out, dict) else str(out)

    if m == "bocpd":
        _ensure_bocpd_available()
        out = stable_regime_analysis(
            assets,
            method="bocpd",
            start_date=start_date,
            end_date=end_date,
            return_result=True,
            verbose=verbose,
            **kwargs,
        )
        return str(out.get("report", "")) if isinstance(out, dict) else str(out)

    raise ValueError(f"Unknown method: {method!r}. Expected 'hmm' or 'bocpd'.")

# Friendly aliases
analyze = stable_regime_analysis
report = stable_report