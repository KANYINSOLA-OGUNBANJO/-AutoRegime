# autoregime/engines/bocpd.py
from __future__ import annotations

"""
BOCPD-like engine (self-contained)

- No imports from hmm_sticky (avoids circular/private helper issues).
- Simple online change detection (CUSUM-style) with a "hazard" knob
  (lower hazard => fewer switches), plus min_segment_days post-filter.
- Produces the same schema as HMM engine:
    {"report": str, "regime_timeline": list[dict], "meta": dict}
"""

from dataclasses import dataclass
from typing import Any, Optional, Dict, List

import numpy as np
import pandas as pd

# Optional dependency only needed for ticker strings
try:  # pragma: no cover
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # noqa: N816

TRADING_DAYS = 252


# ---------------- Utilities (local on purpose) ----------------
def _annualize_return(daily_returns: pd.Series) -> float:
    return float(daily_returns.mean() * TRADING_DAYS)


def _annualize_vol(daily_returns: pd.Series) -> float:
    return float(daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS))


def _max_drawdown_from_prices(prices: pd.Series) -> float:
    cummax = prices.cummax()
    return float((prices / cummax - 1.0).min())


def _winsorize(s: pd.Series, p=0.005) -> pd.Series:
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def _compute_returns(prices: pd.Series) -> pd.Series:
    r = np.log(prices).diff().dropna()
    r.name = "logret"
    return r


def _min_segment_enforce(states: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1:
        return states
    s = states.copy()
    n = len(s)

    # Forward pass
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        run_len = j - i + 1
        if run_len < min_len:
            left = s[i - 1] if i > 0 else None
            right = s[j + 1] if j + 1 < n else None
            target = right if right is not None else (left if left is not None else s[i])
            s[i : j + 1] = target
        i = j + 1

    # Backward clean-up
    i = n - 1
    while i >= 0:
        j = i
        while j - 1 >= 0 and s[j - 1] == s[i]:
            j -= 1
        run_len = i - j + 1
        if run_len < min_len:
            left = s[j - 1] if j - 1 >= 0 else None
            right = s[i + 1] if i + 1 < n else None
            target = left if left is not None else (right if right is not None else s[i])
            s[j : i + 1] = target
        i = j - 1
    return s


def _label_states_rich(returns: pd.Series, states: np.ndarray) -> dict[int, str]:
    df = pd.DataFrame({"state": states}, index=returns.index).join(returns.rename("r"))
    g = df.groupby("state")["r"]
    stats = pd.DataFrame({"mu": g.mean(), "sigma": g.std(ddof=0).replace(0, np.nan)})
    stats["sharpe"] = stats["mu"] / stats["sigma"]

    labels: dict[int, str] = {}
    remaining = set(stats.index)

    # Goldilocks: pos mu, highest Sharpe
    pos = stats[stats["mu"] > 0].sort_values("sharpe", ascending=False)
    if not pos.empty and np.isfinite(pos["sharpe"].iloc[0]):
        gold = pos.index[0]
        labels[gold] = "Goldilocks"
        remaining.discard(gold)

    # Bull: highest mu among remaining positives
    rem_pos = stats.loc[list(remaining)]
    rem_pos = rem_pos[rem_pos["mu"] > 0]
    if not rem_pos.empty:
        bull = rem_pos["mu"].idxmax()
        labels[bull] = "Bull Market"
        remaining.discard(bull)

    # Bear: lowest mu overall
    if remaining:
        bear = stats.loc[list(remaining)]["mu"].idxmin()
        labels[bear] = "Bear Market"
        remaining.discard(bear)

    # Sideways: closest to 0 mu and not too volatile
    if remaining:
        rem = stats.loc[list(remaining)].copy()
        rem["abs_mu"] = np.abs(rem["mu"])
        med_sigma = stats["sigma"].median()
        cand = rem[rem["sigma"] <= med_sigma] if np.isfinite(med_sigma) else rem
        if cand.empty:
            cand = rem
        side = cand.sort_values(["abs_mu", "sigma"], ascending=[True, True]).index[0]
        labels[side] = "Sideways"
        remaining.discard(side)

    # Risk-Off: negative mu, highest sigma (or just highest sigma)
    if remaining:
        rem = stats.loc[list(remaining)]
        cand = rem[rem["mu"] <= 0]
        if cand.empty:
            cand = rem
        risk = cand["sigma"].idxmax()
        labels[risk] = "Risk-Off"
        remaining.discard(risk)

    for st in list(remaining):
        labels[st] = "Correction" if stats.loc[st, "mu"] < 0 else "Neutral"
    return labels


def _build_timeline(
    index: pd.DatetimeIndex,
    states: np.ndarray,
    returns: pd.Series,
    prices: pd.Series,
    state_labels: dict[int, str],
) -> pd.DataFrame:
    df = pd.DataFrame({"date": index, "state": states}).set_index("date")
    df["label"] = df["state"].map(state_labels)

    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    cur_state, start, prev_dt = None, None, None
    for dt, row in df.iterrows():
        st = int(row["state"])
        if cur_state is None:
            cur_state, start = st, dt
        elif st != cur_state:
            runs.append((start, prev_dt, cur_state))
            cur_state, start = st, dt
        prev_dt = dt
    if start is not None and prev_dt is not None:
        runs.append((start, prev_dt, cur_state))

    out = []
    for i, (s, e, st) in enumerate(runs, 1):
        mask = (returns.index >= s) & (returns.index <= e)
        r_seg = returns.loc[mask]
        p_seg = prices.loc[mask]
        ann_ret = _annualize_return(r_seg)
        ann_vol = _annualize_vol(r_seg)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        mdd = _max_drawdown_from_prices(p_seg)
        out.append(
            {
                "period_index": i,
                "label": state_labels.get(st, f"Regime {st}"),
                "state": int(st),
                "start": s.date().isoformat(),
                "end": e.date().isoformat(),
                "trading_days": int(mask.sum()),
                "years": float(mask.sum() / TRADING_DAYS),
                "ann_return": float(ann_ret),
                "ann_vol": float(ann_vol),
                "sharpe": float(sharpe),
                "max_drawdown": float(mdd),
            }
        )
    return pd.DataFrame(out)


# ---------------- Data loading ----------------
def _load_prices(assets: Any, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    if isinstance(assets, str):
        if yf is None:
            raise RuntimeError("yfinance is required. Install with: pip install yfinance")
        try:
            df = yf.download(assets, start=start_date, end=end_date, auto_adjust=True, progress=False)
        except Exception as e:
            raise RuntimeError(f"Failed to download data for {assets} via yfinance: {e}") from e
        if df is None or df.empty:
            raise ValueError(f"No data returned for {assets}.")
        close = df["Close"]
        ser = close.iloc[:, 0].astype(float) if isinstance(close, pd.DataFrame) else close.astype(float)
        ser = ser.dropna()
        ser.name = str(assets)
        return pd.DataFrame(ser)

    if isinstance(assets, pd.Series):
        name = getattr(assets, "name", None) or "asset"
        ser = assets.astype(float).dropna()
        return ser.to_frame(name=name)

    if isinstance(assets, pd.DataFrame):
        if assets.shape[1] < 1:
            raise ValueError("Price DataFrame has no columns.")
        return assets.astype(float).dropna(how="all")

    raise TypeError("assets must be a ticker str, pandas Series, or DataFrame of prices")


# ---------------- Simple online change detector (CUSUM-ish) ----------------
def _detect_changes(
    r: pd.Series,
    *,
    roll: int = 25,
    hazard: float = 1 / 90.0,  # lower => fewer switches
    z_floor: float = 1.6,
    z_ceiling: float = 3.2,
) -> np.ndarray:
    """Return 0/1 flags; 1 marks a change point. Lower hazard => higher threshold => fewer changes."""
    r = r.copy()
    mu = r.rolling(roll, min_periods=roll).mean().shift(1)
    sd = r.rolling(roll, min_periods=roll).std(ddof=0).shift(1).replace(0, np.nan)
    z = (r - mu) / sd
    z = z.fillna(0.0).abs()

    h = float(max(1e-6, min(0.2, hazard)))
    z_thr = float(z_ceiling - (z_ceiling - z_floor) * (h / 0.2))  # h=0.2 => z_floor; h~0 => z_ceiling

    flags = (z > z_thr).astype(int).values
    flags[: roll * 2] = 0  # warm-up
    return flags


def _states_from_flags(flags: np.ndarray) -> np.ndarray:
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
    hazard: float = 1 / 90.0,       # lower => fewer switches
    min_segment_days: int = 20,     # post-filter tiny segments
    return_result: bool = True,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Simple BOCPD-like analysis. The "hazard" parameter tunes sensitivity.
    """
    # Load & prepare
    px = _load_prices(assets, start_date, end_date)
    if px.shape[1] != 1:
        px = px.iloc[:, [0]]
    prices = px.iloc[:, 0].dropna().astype(float)

    # Data availability for meta
    try:
        data_earliest = str(pd.Timestamp(prices.index.min()).date())
        data_latest = str(pd.Timestamp(prices.index.max()).date())
    except Exception:
        data_earliest = ""
        data_latest = ""

    r = _compute_returns(prices)
    r_w = _winsorize(r, p=0.005)

    # Detect changes
    flags = _detect_changes(r_w, hazard=hazard)
    raw_states = _states_from_flags(flags)

    # Enforce min segment length
    states = _min_segment_enforce(raw_states, min_segment_days)

    # Label states
    labels = _label_states_rich(r_w.iloc[: len(states)], states)

    # Build timeline
    tl = _build_timeline(
        r_w.index[: len(states)],
        states,
        r_w,
        prices.iloc[1 : len(states) + 1],
        labels,
    )

    # Report
    lines = ["REGIME ANALYSIS (BOCPD-like)", "==============================", ""]
    for _, row in tl.iterrows():
        lines += [
            f"PERIOD {int(row['period_index'])}: {row['label']}",
            f"   Duration: {row['start']} to {row['end']}",
            f"   Length: {int(row['trading_days'])} trading days ({row['years']:.1f} years)",
            f"   Annual Return: {row['ann_return']*100:.1f}%",
            f"   Annual Volatility: {row['ann_vol']*100:.1f}%",
            f"   Sharpe Ratio: {row['sharpe']:.2f}",
            f"   Max Drawdown: {row['max_drawdown']*100:.1f}%",
            "",
        ]
    if not tl.empty:
        last = tl.iloc[-1]
        lines += [
            "CURRENT MARKET STATUS:",
            f"   Active Regime: {last['label']}",
            f"   Regime Started: {last['start']}",
            f"   Duration So Far: {int(last['trading_days'])} days",
        ]
    report = "\n".join(lines)

    meta = {
        "method": "bocpd_simple",
        "hazard": float(hazard),
        "min_segment_days": int(min_segment_days),
        "n_obs": int(len(r_w)),
        "data_range": {"earliest": data_earliest, "latest": data_latest},
    }

    if return_result:
        return {
            "report": report,
            "regime_timeline": tl.to_dict(orient="records"),
            "meta": meta,
        }
    return {"report": report, "timeline": tl, "meta": meta}