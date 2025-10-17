# reliable_regime.py
from __future__ import annotations

from typing import Optional, Dict, Any, Union
import os
import numpy as np
import pandas as pd
import warnings

import autoregime as ar  # your package


SeriesLike = Union[pd.Series, pd.DataFrame, str]


# ---------- helpers ----------

def _synthetic_prices(
    start: str = "2020-01-01",
    n_days: int = 1500,
    seed: Optional[int] = None,
    s0: float = 100.0,
    mu: float = 0.08,   # annual drift
    sigma: float = 0.20 # annual vol
) -> pd.Series:
    """Simple GBM path for offline tests (business days only)."""
    rng = np.random.default_rng(None if seed is None else int(seed))
    dates = pd.bdate_range(start=start, periods=n_days, freq="B")
    dt = 1.0 / 252.0
    z = rng.standard_normal(len(dates))
    r = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z
    px = s0 * np.exp(np.cumsum(r))
    return pd.Series(px, index=dates, name="SYN").astype(float)


def _clean_prices(px: pd.Series) -> pd.Series:
    """Monotonic index, finite, strictly positive, no gaps/dupes."""
    if not isinstance(px, pd.Series):
        raise TypeError("Expected a pd.Series of prices.")

    s = px.copy()

    # sort, drop duplicate index
    s = s[~s.index.duplicated(keep="first")].sort_index()

    # ensure dtype & finites
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # enforce positivity (log-returns downstream)
    s = s.where(s > 0)

    # fill tiny holes at ends if any, then drop residual NaNs
    s = s.ffill().bfill()
    s = s.dropna()

    # add microscopic jitter to avoid exact-constant windows that can kill covariances
    if len(s) >= 2:
        tiny = 1e-10
        noise = (np.random.default_rng().standard_normal(len(s)) * tiny)
        s = s * (1.0 + noise)

    if len(s) < 30:
        raise ValueError("Not enough observations after cleaning (need >= 30).")

    return s


def _tag_retry(res: Dict[str, Any], attempt: int, note: str) -> Dict[str, Any]:
    """Attach a retry/debug note into meta.notes."""
    try:
        meta = dict(res.get("meta", {}))
        notes = dict(meta.get("notes", {}))
        notes["robust_retry"] = {"attempt": attempt, "note": note}
        meta["notes"] = notes
        res["meta"] = meta
    except Exception:
        pass
    return res


def _call_analysis(
    asset: SeriesLike,
    *,
    method: str,
    start_date: Optional[str],
    end_date: Optional[str],
    random_state: int,
    min_segment_days: int,
    sticky: Optional[float] = None,
    k_floor: Optional[int] = None,
    k_cap: Optional[int] = None,
    conservative: bool = True,
) -> Dict[str, Any]:
    """Single guarded call into ar.stable_regime_analysis."""
    engine_kwargs: Dict[str, Any] = {"min_segment_days": int(min_segment_days)}
    if method.lower() == "hmm":
        if sticky is None:
            sticky = 0.985 if conservative else 0.98
        engine_kwargs["sticky"] = float(sticky)
        # If your engine supports these, they’re passed through; otherwise safely ignored
        if k_floor is not None:
            engine_kwargs["k_floor"] = int(k_floor)
        if k_cap is not None:
            engine_kwargs["k_cap"] = int(k_cap)

    return ar.stable_regime_analysis(
        asset,
        method=method,
        start_date=start_date,
        end_date=end_date,
        return_result=True,
        verbose=False,
        random_state=int(random_state),
        **engine_kwargs,
    )


# ---------- public API ----------

def reliable_regime_analysis(
    symbol: str = "SPY",
    *,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    method: str = "hmm",
    min_segment_days: int = 5,
    seed: Optional[int] = None,
    conservative: bool = True,
) -> Dict[str, Any]:
    """
    Hardened wrapper for tests:
    - Accepts `seed` (controls RNG & HMM random_state).
    - Tries live data first unless AUTOREGIME_OFFLINE_TEST=1.
    - Cleans data and retries HMM with safer configs if it hits NaN/Inf/singular covariances.
    - As a last resort uses synthetic prices.

    Returns the same dict schema as ar.stable_regime_analysis(..., return_result=True).
    """
    # seeding for reproducibility
    if seed is not None:
        try:
            np.random.seed(int(seed))
        except Exception:
            pass
    base_state = 42 if seed is None else int(seed)

    offline_env = os.environ.get("AUTOREGIME_OFFLINE_TEST", "0") == "1"

    # ---- 1) Try live symbol unless explicitly offline
    if not offline_env:
        try:
            res = _call_analysis(
                symbol,
                method=method,
                start_date=start,
                end_date=end,
                random_state=base_state,
                min_segment_days=min_segment_days,
                sticky=0.985 if method.lower() == "hmm" else None,
                conservative=conservative,
            )
            return res
        except Exception as e:
            # fall through to offline path
            warnings.warn(f"Live data path failed ({type(e).__name__}): {e}. Falling back to offline synthetic.")

    # ---- 2) Offline synthetic with robust retries
    px = _clean_prices(_synthetic_prices(start=start, n_days=1500, seed=seed, s0=100.0))

    # Attempt A: baseline conservative HMM
    try:
        resA = _call_analysis(
            px,
            method=method,
            start_date=None,
            end_date=None,
            random_state=base_state,
            min_segment_days=min_segment_days,
            sticky=0.985 if method.lower() == "hmm" else None,
            conservative=conservative,
        )
        # mark offline
        resA = _tag_retry(resA, attempt=0, note="baseline")
        meta = dict(resA.get("meta", {}))
        notes = dict(meta.get("notes", {}))
        notes["offline_synthetic"] = True
        meta["notes"] = notes
        resA["meta"] = meta
        return resA
    except Exception as eA:
        msgA = f"{type(eA).__name__}: {eA}"

    # Attempt B: safer HMM (smaller K range, stickier, new seed)
    try:
        resB = _call_analysis(
            px,
            method=method,
            start_date=None,
            end_date=None,
            random_state=base_state + 7,
            min_segment_days=max(6, min_segment_days),
            sticky=0.99 if method.lower() == "hmm" else None,
            k_floor=2,
            k_cap=4,
            conservative=True,
        )
        resB = _tag_retry(resB, attempt=1, note="retry: sticky=0.99, k_floor=2, k_cap=4, +7 seed")
        meta = dict(resB.get("meta", {}))
        notes = dict(meta.get("notes", {}))
        notes["offline_synthetic"] = True
        meta["notes"] = notes
        resB["meta"] = meta
        return resB
    except Exception as eB:
        msgB = f"{type(eB).__name__}: {eB}"

    # Attempt C: extra-safe HMM (even smaller K cap & different seed)
    try:
        resC = _call_analysis(
            px,
            method=method,
            start_date=None,
            end_date=None,
            random_state=base_state + 101,
            min_segment_days=max(8, min_segment_days),
            sticky=0.992 if method.lower() == "hmm" else None,
            k_floor=2,
            k_cap=3,
            conservative=True,
        )
        resC = _tag_retry(resC, attempt=2, note="retry: sticky=0.992, k_floor=2, k_cap=3, +101 seed")
        meta = dict(resC.get("meta", {}))
        notes = dict(meta.get("notes", {}))
        notes["offline_synthetic"] = True
        meta["notes"] = notes
        resC["meta"] = meta
        return resC
    except Exception as eC:
        msgC = f"{type(eC).__name__}: {eC}"

    # If everything failed, surface a concise error with context
    raise RuntimeError(
        "HMM robustness retries exhausted. Last errors were:\n"
        f"  A) {msgA}\n"
        f"  B) {msgB}\n"
        f"  C) {msgC}\n"
        "Consider checking your engine for NaN/Inf handling and min-covar guards."
    )


if __name__ == "__main__":
    # local smoke test (won't run during pytest collection)
    out = reliable_regime_analysis("SPY", seed=0, min_segment_days=5)
    print(out.get("report", "")[:600], "...\n")
    tl = pd.DataFrame(out.get("regime_timeline", []))
    print(tl.head(3))