# autoregime/engines/bocpd.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

# Optional data source
try:  # pragma: no cover
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # noqa: N816

# Unified reporting helpers (with dynamic risk-free + shock notes)
from ..reporting.common import (
    TRADING_DAYS,
    MIN_CAGR_DAYS,
    ensure_positive_prices,
    compute_log_returns,
    winsorize,
    build_timeline_from_state_runs,
    format_report,
    get_daily_risk_free,
    compute_excess_log_returns,
    annotate_intra_regime_shocks,
)

# Optional timeline validator (non-blocking if missing)
try:
    from ..core.validation import validate_timeline  # type: ignore
    _HAS_VALIDATOR = True
except Exception:
    _HAS_VALIDATOR = False


# ---------------- Segment/State utilities ----------------
def _runs_from_states(states: np.ndarray) -> List[Tuple[int, int, int]]:
    runs: List[Tuple[int, int, int]] = []
    if len(states) == 0:
        return runs
    i = 0
    n = len(states)
    while i < n:
        j = i
        while j + 1 < n and states[j + 1] == states[i]:
            j += 1
        runs.append((i, j, int(states[i])))
        i = j + 1
    return runs


def _min_segment_enforce(states: np.ndarray, min_len: int) -> np.ndarray:
    """Merge too-short runs into neighbors UNTIL none remain (prefer longer neighbor; ties → right)."""
    if min_len <= 1 or len(states) == 0:
        return states
    s = states.copy().astype(int)
    changed = True
    while changed:
        changed = False
        runs = _runs_from_states(s)
        for (a, b, _st), idx in zip(runs, range(len(runs))):
            run_len = b - a + 1
            if run_len >= min_len:
                continue
            left_state = runs[idx - 1][2] if idx - 1 >= 0 else None
            right_state = runs[idx + 1][2] if idx + 1 < len(runs) else None
            left_len = (runs[idx - 1][1] - runs[idx - 1][0] + 1) if left_state is not None else -1
            right_len = (runs[idx + 1][1] - runs[idx + 1][0] + 1) if right_state is not None else -1
            if right_len >= left_len and right_state is not None:
                s[a : b + 1] = right_state
            elif left_state is not None:
                s[a : b + 1] = left_state
            else:
                continue
            changed = True
            break
    return s


def _label_states_rich(returns: pd.Series, states: np.ndarray) -> dict[int, str]:
    """
    Map states to business-friendly labels using μ (mean), σ (vol), and Sharpe.
    Uses the provided `returns` (we pass *excess* raw returns for best semantics).
    """
    df = pd.DataFrame({"state": states}, index=returns.index).join(returns.rename("r"))
    g = df.groupby("state")["r"]
    stats = pd.DataFrame({"mu": g.mean(), "sigma": g.std(ddof=0).replace(0, np.nan)})
    stats["sharpe"] = stats["mu"] / stats["sigma"]

    labels: dict[int, str] = {}
    remaining = set(stats.index)
    pos_idx = stats.index[stats["mu"] > 0].tolist()
    neg_idx = stats.index[stats["mu"] < 0].tolist()

    # Goldilocks: highest Sharpe among μ>0
    if len(pos_idx) > 0:
        pos_stats = stats.loc[pos_idx].sort_values("sharpe", ascending=False)
        if pos_stats["sharpe"].notna().any():
            gld = int(pos_stats.index[0])
            labels[gld] = "Goldilocks"
            remaining.discard(gld)

    # Bear: most negative μ (only if μ<0 exists)
    if len(neg_idx) > 0:
        br = int(stats.loc[neg_idx, "mu"].idxmin())
        labels[br] = "Bear Market"
        remaining.discard(br)

    # Bull: best μ among remaining μ>0
    rem_pos = [i for i in remaining if stats.loc[i, "mu"] > 0]
    if len(rem_pos) > 0:
        bull = int(stats.loc[rem_pos, "mu"].idxmax())
        labels[bull] = "Bull Market"
        remaining.discard(bull)

    # Sideways: |μ| near 0 with low σ
    if len(remaining) > 0:
        rem = stats.loc[list(remaining)].copy()
        rem["abs_mu"] = rem["mu"].abs()
        med_sigma = float(stats["sigma"].median())
        cand = rem[rem["sigma"] <= med_sigma] if np.isfinite(med_sigma) else rem
        side = int(cand.sort_values(["abs_mu", "sigma"], ascending=[True, True]).index[0])
        labels[side] = "Sideways"
        remaining.discard(side)

    # Leftovers
    for st in list(remaining):
        labels[int(st)] = "Steady Growth" if stats.loc[st, "mu"] > 0 else "Risk-Off"

    return labels


# ---------------- Data loading ----------------
def _load_prices(assets: Any, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    if isinstance(assets, str):
        if yf is None:
            raise RuntimeError("yfinance is required. Install with: pip install yfinance")
        df = yf.download(assets, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise ValueError(f"No data returned for {assets}.")
        close = df["Close"]
        ser = (close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close).astype(float)
        ser = ensure_positive_prices(ser)
        ser.name = str(assets)
        return pd.DataFrame(ser)

    if isinstance(assets, pd.Series):
        name = getattr(assets, "name", None) or "asset"
        ser = ensure_positive_prices(assets)
        return ser.to_frame(name=name)

    if isinstance(assets, pd.DataFrame):
        if assets.shape[1] < 1:
            raise ValueError("Price DataFrame has no columns.")
        first = assets.columns[0]
        df = ensure_positive_prices(assets[first]).to_frame(name=str(first))
        return df

    raise TypeError("assets must be a ticker str, pandas Series, or DataFrame of prices")


# ---------------- BOCPD-ish change detector ----------------
def _detect_changes(
    r: pd.Series,
    *,
    roll: int = 25,
    hazard: float = 1 / 90.0,
    z_floor: float = 1.6,
    z_ceiling: float = 3.2,
    use_ewm: bool = True,
    span: int = 20,
) -> np.ndarray:
    """
    Flag points where |z| exceeds a hazard-tuned threshold.
    - If use_ewm: use EWMA mean/std (shifted) for adaptive z.
    - Else: use rolling window mean/std (shifted).
    """
    r = pd.to_numeric(r, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if use_ewm:
        mu = r.ewm(span=span, adjust=False, min_periods=span).mean().shift(1)
        sd = r.ewm(span=span, adjust=False, min_periods=span).std().shift(1).replace(0, np.nan)
        warmup = span * 2
    else:
        mu = r.rolling(roll, min_periods=roll).mean().shift(1)
        sd = r.rolling(roll, min_periods=roll).std(ddof=0).shift(1).replace(0, np.nan)
        warmup = roll * 2

    z = ((r - mu) / sd).abs().fillna(0.0)

    # Hazard-rate → threshold schedule (higher hazard = lower threshold)
    h = float(max(1e-6, min(0.2, hazard)))
    z_thr = float(z_ceiling - (z_ceiling - z_floor) * (h / 0.2))

    flags = (z > z_thr).astype(int).to_numpy()
    flags[: warmup] = 0  # suppress early noise
    return flags


def _states_from_flags(flags: np.ndarray) -> np.ndarray:
    """Convert change flags into monotonically increasing segment IDs."""
    states = np.zeros_like(flags, dtype=int)
    cur = 0
    for i in range(1, len(flags)):
        if flags[i] == 1:
            cur += 1
        states[i] = cur
    return states


# ---------------- Public entrypoint ----------------
@dataclass
class AnalysisResult:
    report: str
    timeline: pd.DataFrame
    meta: dict


def bocpd_regime_analysis(
    assets: Any,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    hazard: float = 1 / 90.0,
    min_segment_days: int = 20,
    show_cagr: bool = False,   # computed in timeline; hidden from report by default
    return_result: bool = True,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    BOCPD-like regime analysis with unified output schema.

    IMPORTANT: We use winsorized returns ONLY for detection; timeline metrics
    (Sharpe/vol) are computed on RAW **excess** returns (r - rf).
    """
    # Load prices
    px = _load_prices(assets, start_date, end_date)
    if px.shape[1] != 1:
        px = px.iloc[:, [0]]
    prices = ensure_positive_prices(px.iloc[:, 0].dropna().astype(float))

    # Data range meta
    data_earliest = str(pd.Timestamp(prices.index.min()).date()) if len(prices) else ""
    data_latest   = str(pd.Timestamp(prices.index.max()).date()) if len(prices) else ""

    # Raw daily log-returns (base series)
    r = compute_log_returns(prices)  # index aligned to P_t
    if len(r) < max(60, min_segment_days * 3):
        report = (
            "REGIME ANALYSIS (BOCPD-like)\n==============================\n\n"
            "Insufficient data to segment reliably."
        )
        meta = {
            "method": "bocpd_simple",
            "hazard": float(hazard),
            "min_segment_days": int(min_segment_days),
            "n_obs": int(len(r)),
            "data_range": {"earliest": data_earliest, "latest": data_latest},
            "notes": {"cagr_suppressed_below_days": MIN_CAGR_DAYS, "show_cagr": bool(show_cagr)},
        }
        return {"report": report, "regime_timeline": [], "meta": meta} if return_result else report

    # Winsorize ONLY for detection robustness
    r_w = winsorize(r, p=0.005)
    r_w = pd.to_numeric(r_w, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if r_w.nunique() <= 1:
        r_w = r_w + 1e-8  # ensure some variance for z-scores

    # Detect changes on winsorized series
    flags = _detect_changes(r_w, hazard=hazard, use_ewm=True, span=20)
    raw_states = _states_from_flags(flags)
    states = _min_segment_enforce(raw_states, min_segment_days)

    # Risk-free (for labeling + timeline Sharpe on RAW excess returns)
    try:
        rf_daily = get_daily_risk_free(r.index.min(), r.index.max(), index=r.index, series="GS10", mode="cc")
    except Exception:
        rf_daily = pd.Series(0.0, index=r.index, name="rf_daily", dtype=float)

    rx_raw = compute_excess_log_returns(r, rf_daily)  # raw excess returns for labels & metrics

    # Labels derived from raw EXCESS returns for business semantics
    labels = _label_states_rich(rx_raw.iloc[: len(states)], states)

    # Timeline (align prices to returns index = P_t)
    px_aligned = prices.reindex(r.index).dropna()
    tl = build_timeline_from_state_runs(
        index=r.index[: len(states)],
        states=states,
        returns=r,  # <-- RAW returns for metric computation (excess handled inside builder)
        prices_aligned_to_returns=px_aligned.iloc[: len(states)],
        state_to_label=labels,
    )

    # Intra-regime shock notes (optional; uses RAW returns)
    tl = annotate_intra_regime_shocks(tl, returns=r)

    # Validate timeline (non-blocking)
    if _HAS_VALIDATOR:
        try:
            v = validate_timeline(tl, min_segment_days=min_segment_days, min_cagr_days=MIN_CAGR_DAYS)
            validation_meta = {"ok": v.ok, "errors": v.errors, "warnings": v.warnings, "info": v.info}
        except Exception as ex:
            validation_meta = {"ok": False, "errors": [f"validator exception: {ex}"], "warnings": [], "info": {}}
    else:
        validation_meta = {"ok": True, "errors": [], "warnings": ["validator_not_installed"], "info": {}}

    # Report — CAGR line hidden by default (dashboard-safe)
    report = format_report(tl, show_cagr=show_cagr, hide_cagr_line=True, title="REGIME ANALYSIS (BOCPD-like)")

    # Meta
    meta = {
        "method": "bocpd_simple",
        "hazard": float(hazard),
        "min_segment_days": int(min_segment_days),
        "states_present": int(len(set(states.tolist()))),
        "n_obs": int(len(r)),
        "data_range": {"earliest": data_earliest, "latest": data_latest},
        "rf": {"series": "GS10", "mode": "cc"},
        "notes": {"cagr_suppressed_below_days": MIN_CAGR_DAYS, "show_cagr": bool(show_cagr)},
        "validation": validation_meta,
    }

    if return_result:
        return {"report": report, "regime_timeline": tl.to_dict(orient="records"), "meta": meta}
    else:
        return report


__all__ = ["bocpd_regime_analysis", "AnalysisResult"]