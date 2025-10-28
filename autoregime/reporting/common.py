# autoregime/reporting/common.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_CAGR_DAYS = 90  # suppress CAGR for segments younger than this


# ---------- internal helpers ----------
def _to_datetime_index(x) -> pd.DatetimeIndex:
    if isinstance(x, pd.DatetimeIndex):
        return x
    return pd.to_datetime(x)


def _as_business_days(start, end) -> pd.DatetimeIndex:
    return pd.bdate_range(pd.to_datetime(start), pd.to_datetime(end))


def _make_finite(s: pd.Series | list | np.ndarray | None, fill: float = 0.0) -> pd.Series:
    """Coerce to float Series and replace inf/NaN with `fill` (default 0.0)."""
    if s is None:
        return pd.Series([], dtype=float)
    s = pd.to_numeric(pd.Series(s), errors="coerce").astype(float)
    return s.replace([np.inf, -np.inf], np.nan).fillna(fill)


# ---------- back-compat aliases some engines import ----------
def ensure_finite_series(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return _make_finite(s, fill=fill)


def sanitize_returns(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return _make_finite(s, fill=fill)


def sanitize_prices(s: pd.Series) -> pd.Series:
    """Legacy alias (kept so `from ..reporting.common import sanitize_prices` never breaks)."""
    if s is None:
        return pd.Series([], dtype=float)
    s = pd.to_numeric(pd.Series(s), errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


# ---------- prices ----------
def ensure_positive_prices(prices: pd.Series, *, min_positive: float = 1e-12) -> pd.Series:
    """Drop non-positive and non-finite observations; keep index & name."""
    if prices is None or len(prices) == 0:
        return pd.Series([], dtype=float)
    p = pd.to_numeric(prices, errors="coerce").astype(float)
    p = p.where(p > min_positive, np.nan)
    p = p.replace([np.inf, -np.inf], np.nan).dropna()
    p.name = getattr(prices, "name", None) or "price"
    return p


# ---------- risk-free (dynamic) ----------
_RF_CACHE: Dict[Tuple[str, str, str, str], pd.Series] = {}


def get_daily_risk_free(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    series: str = "GS10",
    source: str = "fred",
    mode: str = "cc",  # "cc" (continuous) or "simple"
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    Fetch annualized risk-free (GS10) from FRED, convert to daily
    (continuous comp if mode='cc'; else simple), align to `index`
    (or business days between start/end). Falls back to 0.0 on error.
    """
    start_iso = pd.to_datetime(start).date().isoformat()
    end_iso = pd.to_datetime(end).date().isoformat()
    key = (start_iso, end_iso, series, mode)

    if key in _RF_CACHE:
        rf = _RF_CACHE[key]
    else:
        try:
            from pandas_datareader import data as pdr  # type: ignore
            ser = pdr.DataReader(series, source, start_iso, end_iso)[series].astype(float) / 100.0  # annual decimal
            ser = ser.sort_index()
        except Exception:
            # Offline or FRED unavailable → zero RF; engines keep running
            ser = pd.Series(0.0, index=_as_business_days(start_iso, end_iso), name=series, dtype=float)

        if mode == "cc":
            daily = np.log1p(ser / TRADING_DAYS)
        else:
            daily = ser / TRADING_DAYS
        rf = pd.Series(daily, name="rf_daily").astype(float)
        rf = _make_finite(rf, fill=0.0)
        _RF_CACHE[key] = rf

    idx = _to_datetime_index(index) if index is not None else _as_business_days(start_iso, end_iso)
    rf_aligned = rf.reindex(idx).ffill().fillna(0.0)
    rf_aligned.name = "rf_daily"
    return rf_aligned


# Alias some engines might import
get_risk_free_daily = get_daily_risk_free


def compute_excess_log_returns(daily_log_returns: pd.Series, rf_daily: pd.Series) -> pd.Series:
    """excess = raw daily log-returns - rf_daily (aligned, finite)."""
    if daily_log_returns is None or len(daily_log_returns) == 0:
        return pd.Series([], dtype=float)
    r = _make_finite(daily_log_returns, fill=np.nan)
    rf = _make_finite(rf_daily, fill=0.0)
    rx = (r.reindex_like(rf) - rf).replace([np.inf, -np.inf], np.nan).dropna()
    rx.name = "excess_logret"
    return rx


# ---------- stats helpers ----------
def annualize_return_mean(daily_log_returns: pd.Series) -> float:
    if daily_log_returns is None or len(daily_log_returns) == 0:
        return float("nan")
    return float(_make_finite(daily_log_returns).mean() * TRADING_DAYS)


def annualize_vol(daily_log_returns: pd.Series) -> float:
    if daily_log_returns is None or len(daily_log_returns) == 0:
        return float("nan")
    return float(_make_finite(daily_log_returns).std(ddof=0) * np.sqrt(TRADING_DAYS))


def max_drawdown_from_prices(prices: pd.Series) -> float:
    if prices is None or len(prices) == 0:
        return float("nan")
    p = _make_finite(prices).replace(0, np.nan).ffill()
    if len(p) == 0 or not np.isfinite(p.iloc[0]):
        return float("nan")
    return float((p / p.cummax() - 1.0).min())


def total_return_from_prices(prices: pd.Series) -> float:
    if prices is None or len(prices) < 2:
        return float("nan")
    p = _make_finite(prices).replace(0, np.nan).dropna()
    if len(p) < 2:
        return float("nan")
    return float(p.iloc[-1] / p.iloc[0] - 1.0)


def cagr_from_prices(prices: pd.Series, trading_days: int) -> float:
    if prices is None or len(prices) < 2 or trading_days <= 0:
        return float("nan")
    p = _make_finite(prices).replace(0, np.nan).dropna()
    if len(p) < 2:
        return float("nan")
    total = float(p.iloc[-1] / p.iloc[0])
    if total <= 0:
        return float("nan")
    years = trading_days / TRADING_DAYS
    if years <= 0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


def winsorize(s: pd.Series, p: float = 0.005) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series([], dtype=float)
    s = pd.to_numeric(s, errors="coerce").astype(float)
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def compute_log_returns(prices: pd.Series) -> pd.Series:
    if prices is None or len(prices) == 0:
        return pd.Series([], dtype=float)
    p = ensure_positive_prices(prices)
    r = np.log(p).diff().replace([np.inf, -np.inf], np.nan).dropna()
    r.name = "logret"
    return r


# ---------- intra-regime shock annotation ----------
def annotate_intra_regime_shocks(
    tl: pd.DataFrame,
    returns: Optional[pd.Series] = None,
    *,
    z_threshold: float = 3.0,
    ewm_span: int = 20,
) -> pd.DataFrame:
    if tl is None or tl.empty or returns is None or len(returns) == 0:
        return tl

    r = _make_finite(returns, fill=np.nan).dropna()
    if len(r) == 0:
        return tl

    mu = r.ewm(span=ewm_span, adjust=False, min_periods=ewm_span).mean()
    sd = r.ewm(span=ewm_span, adjust=False, min_periods=ewm_span).std().replace(0, np.nan)
    z = ((r - mu) / sd).abs().fillna(0.0)
    shock_idx = z[z > float(z_threshold)].index

    if len(shock_idx) == 0:
        return tl

    out = tl.copy()
    for i, row in out.iterrows():
        try:
            s = pd.to_datetime(row.get("start", None))
            e = pd.to_datetime(row.get("end", None))
        except Exception:
            continue
        if pd.isna(s) or pd.isna(e):
            continue
        n = int(((shock_idx >= s) & (shock_idx <= e)).sum())
        if n > 0:
            prev = str(row.get("note") or "").strip()
            tag = f"intra_shock(|z|>{z_threshold:.1f}, n={n})"
            out.at[i, "note"] = (prev + (", " if prev else "") + tag)
    return out


# ---------- generic timeline builder ----------
def build_timeline_from_state_runs(
    *,
    index: pd.DatetimeIndex,
    states: np.ndarray,
    returns: pd.Series,                      # raw daily log-returns (portfolio)
    prices_aligned_to_returns: pd.Series,    # aligned to returns' index (same dates)
    state_to_label: dict[int, str],
) -> pd.DataFrame:
    def runs_from_states(x: np.ndarray) -> List[tuple[int, int, int]]:
        out: List[tuple[int, int, int]] = []
        if len(x) == 0:
            return out
        i, n = 0, len(x)
        while i < n:
            j = i
            while j + 1 < n and x[j + 1] == x[i]:
                j += 1
            out.append((i, j, int(x[i])))
            i = j + 1
        return out

    if returns is None or len(returns) == 0:
        return pd.DataFrame([])

    full_idx = _to_datetime_index(index)
    try:
        rf_daily = get_daily_risk_free(full_idx.min(), full_idx.max(), index=full_idx, series="GS10", mode="cc")
    except Exception:
        rf_daily = pd.Series(0.0, index=full_idx, name="rf_daily", dtype=float)

    r = pd.to_numeric(returns, errors="coerce").astype(float)
    rx = (r.reindex_like(rf_daily) - rf_daily).replace([np.inf, -np.inf], np.nan).dropna()
    rx.name = "excess_logret"

    rows: List[Dict[str, Any]] = []
    pos_to_dt = list(index)

    for k, (a, b, st) in enumerate(runs_from_states(states), 1):
        s_dt, e_dt = pos_to_dt[a], pos_to_dt[b]
        r_seg_ex = rx.iloc[a : b + 1]
        p_seg = prices_aligned_to_returns.iloc[a : b + 1]

        tdays = int(b - a + 1)
        years = float(tdays / TRADING_DAYS)

        ann_ret_mean = annualize_return_mean(r_seg_ex)
        ann_vol = annualize_vol(r_seg_ex)
        sharpe = float(ann_ret_mean / ann_vol) if np.isfinite(ann_vol) and ann_vol > 0 else float("nan")

        total_ret = total_return_from_prices(p_seg)
        raw_cagr = cagr_from_prices(p_seg, tdays)
        cagr = float(raw_cagr) if (tdays >= MIN_CAGR_DAYS and np.isfinite(raw_cagr)) else float("nan")
        mdd = max_drawdown_from_prices(p_seg)

        p0 = float(p_seg.iloc[0]) if len(p_seg) else float("nan")
        p1 = float(p_seg.iloc[-1]) if len(p_seg) else float("nan")

        rows.append(
            {
                "period_index": k,
                "label": state_to_label.get(st, f"Regime {st}"),
                "state": int(st),
                "start": pd.Timestamp(s_dt).date().isoformat(),
                "end": pd.Timestamp(e_dt).date().isoformat(),
                "trading_days": tdays,
                "years": years,
                "price_start": p0,
                "price_end": p1,
                "price_change_abs": (p1 - p0) if (np.isfinite(p0) and np.isfinite(p1)) else float("nan"),
                "price_change_pct": float(total_ret),
                "period_return": float(total_ret),
                "ann_return": cagr,         # optional; often hidden if < MIN_CAGR_DAYS
                "ann_vol": float(ann_vol),
                "sharpe": float(sharpe) if np.isfinite(sharpe) else float("nan"),
                "max_drawdown": float(mdd),
                "ann_return_mean": float(ann_ret_mean),  # useful for label guards
                "young_segment": bool(tdays < MIN_CAGR_DAYS),
                "note": "",
                "pos_start": int(a),
                "pos_end": int(b),
            }
        )

    return pd.DataFrame(rows)


# ---------- uniform text report ----------
def _fmt_price(x: Optional[float], *, currency_symbol: str = "$") -> str:
    if x is None or not isinstance(x, (int, float)) or not np.isfinite(float(x)):
        return "n/a"
    return f"{currency_symbol}{float(x):,.2f}"


def format_report(
    tl: pd.DataFrame,
    *,
    show_cagr: bool = False,
    hide_cagr_line: bool = True,
    title: str = "REGIME ANALYSIS",
    currency_symbol: str = "$",
    show_price_move: bool = True,
) -> str:
    if tl is None or len(tl) == 0:
        return f"{title}\n{('=' * len(title))}\n\nNo timeline."

    lines = [title, "=" * len(title), ""]
    for _, r in tl.iterrows():
        period = int(r.get("period_index", 0))
        label = str(r.get("label", ""))
        start, end = str(r.get("start", "")), str(r.get("end", ""))
        tdays = int(r.get("trading_days", 0))
        years = float(r.get("years", 0.0))

        pr = float(r.get("period_return", float("nan")))
        cagr = float(r.get("ann_return", float("nan")))
        vol = float(r.get("ann_vol", float("nan")))
        sharpe = float(r.get("sharpe", float("nan")))
        mdd = float(r.get("max_drawdown", float("nan")))
        note = str(r.get("note", "") or "").strip()

        p0 = r.get("price_start", None)
        p1 = r.get("price_end", None)

        lines.append(f"PERIOD {period}: {label}")
        lines.append(f"   Duration: {start} to {end}")
        lines.append(f"   Length: {tdays} trading days ({years:.1f} years)")
        if show_price_move:
            lines.append(
                f"   Price Move: {_fmt_price(p0, currency_symbol=currency_symbol)} → "
                f"{_fmt_price(p1, currency_symbol=currency_symbol)}"
            )
        lines.append(f"   Period Return: {pr * 100:.1f}%")

        if not hide_cagr_line:
            if show_cagr and np.isfinite(cagr):
                lines.append(f"   Annual Return (CAGR): {cagr * 100:.1f}%")
            else:
                lines.append("   Annual Return (CAGR): n/a")

        lines.append(f"   Annual Volatility: {vol * 100:.1f}%" if np.isfinite(vol) else "   Annual Volatility: n/a")
        lines.append(f"   Sharpe Ratio (excess Rf): {sharpe:.2f}" if np.isfinite(sharpe) else "   Sharpe Ratio (excess Rf): n/a")
        lines.append(f"   Max Drawdown: {mdd * 100:.1f}%")
        if note:
            lines.append(f"   Note: {note}")
        lines.append("")

    if not tl.empty:
        last = tl.iloc[-1]
        lines += [
            "CURRENT MARKET STATUS:",
            f"   Active Regime: {last.get('label','')}",
            f"   Regime Started: {last.get('start','')}",
            f"   Duration So Far: {int(last.get('trading_days', 0))} trading days",
        ]
    return "\n".join(lines)


# Back-compat alias
format_regime_report = format_report


__all__ = [
    "TRADING_DAYS",
    "MIN_CAGR_DAYS",
    "winsorize",
    "compute_log_returns",
    "compute_excess_log_returns",
    "get_daily_risk_free",
    "get_risk_free_daily",
    "annualize_return_mean",
    "annualize_vol",
    "max_drawdown_from_prices",
    "total_return_from_prices",
    "cagr_from_prices",
    "build_timeline_from_state_runs",
    "format_report",
    "format_regime_report",
    "ensure_positive_prices",
    "ensure_finite_series",
    "sanitize_returns",
    "sanitize_prices",
    "annotate_intra_regime_shocks",
]