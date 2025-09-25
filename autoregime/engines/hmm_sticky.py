"""
Sticky HMM engine for AutoRegime

- Robust preprocessing (log-returns, winsorize, robust scale)
- Sticky transition prior (diag-heavy init) + Viterbi decode
- Minimum segment length enforcement to stop 1-day flip-flops
- Clean timeline DataFrame + human-readable report
- Works with a ticker (via yfinance) or a price Series/DataFrame

Public entrypoint (compatible):
    stable_regime_analysis(
        assets,
        start_date=None, end_date=None,
        n_components='auto',
        # NEW: auto-K guardrails so we don't collapse to 2 states
        k_floor=4, k_cap=6, auto_k_metric="bic",
        sticky=0.98, min_segment_days=20,
        return_result=True, random_state=42, verbose=True
    )

Convenience:
    stable_report(...) -> str
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

# Optional dependency: only import if a ticker string is passed
try:  # pragma: no cover
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # noqa: N816

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler

TRADING_DAYS = 252


# ---------------- Utilities ----------------
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


@dataclass
class AnalysisResult:
    report: str
    timeline: pd.DataFrame
    meta: dict


# ---------------- Data loading ----------------
def _load_prices(assets: Any, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """Return a price DataFrame with one column if assets is a single ticker/Series."""
    # Ticker string
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
        if isinstance(close, pd.DataFrame):
            if close.shape[1] < 1:
                raise ValueError(f"No Close columns returned for {assets}.")
            ser = close.iloc[:, 0].astype(float)
        else:
            ser = close.astype(float)
        ser = ser.dropna()
        ser.name = str(assets)
        return pd.DataFrame(ser)

    # Pandas Series
    if isinstance(assets, pd.Series):
        name = getattr(assets, "name", None) or "asset"
        ser = assets.astype(float).dropna()
        return ser.to_frame(name=name)

    # Pandas DataFrame
    if isinstance(assets, pd.DataFrame):
        if assets.shape[1] < 1:
            raise ValueError("Price DataFrame has no columns.")
        return assets.astype(float).dropna(how="all")

    raise TypeError("assets must be a ticker str, pandas Series, or DataFrame of prices")


# ---------------- Post-processing helpers ----------------
def _min_segment_enforce(states: np.ndarray, min_len: int) -> np.ndarray:
    """Merge too-short runs into neighbors (greedy, forward then backward)."""
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
    """
    Map hidden states to business-friendly labels using μ (mean), σ (vol), and Sharpe.

      • Goldilocks  : highest Sharpe among positive-μ states
      • Bull Market : highest μ among remaining positive-μ states
      • Bear Market : lowest μ overall
      • Sideways    : |μ| near 0 with low σ
      • Steady Growth: positive μ (not Bull/Goldilocks), moderate σ
      • Risk-Off    : negative μ with high σ (defensive/volatile)

    If there are fewer than 6 states, we assign a sensible subset.
    """
    df = pd.DataFrame({"state": states}, index=returns.index).join(returns.rename("r"))
    g = df.groupby("state")["r"]
    stats = pd.DataFrame(
        {
            "mu": g.mean(),
            "sigma": g.std(ddof=0).replace(0, np.nan),
        }
    )
    stats["sharpe"] = stats["mu"] / stats["sigma"]

    labels: dict[int, str] = {}
    remaining = set(stats.index)

    # 1) Goldilocks: positive μ, highest Sharpe
    pos = stats[stats["mu"] > 0].sort_values("sharpe", ascending=False)
    if not pos.empty and np.isfinite(pos["sharpe"].iloc[0]):
        gold = pos.index[0]
        labels[gold] = "Goldilocks"
        remaining.discard(gold)

    # 2) Bull: highest μ among remaining positives
    rem_pos = stats.loc[list(remaining)]
    rem_pos = rem_pos[rem_pos["mu"] > 0]
    if not rem_pos.empty:
        bull = rem_pos["mu"].idxmax()
        labels[bull] = "Bull Market"
        remaining.discard(bull)

    # 3) Bear: lowest μ overall among remaining
    if remaining:
        bear = stats.loc[list(remaining)]["mu"].idxmin()
        labels[bear] = "Bear Market"
        remaining.discard(bear)

    # 4) Sideways: closest μ to 0 with low σ
    if remaining:
        rem = stats.loc[list(remaining)].copy()
        rem["abs_mu"] = np.abs(rem["mu"])
        med_sigma = stats["sigma"].median()
        cand = rem[rem["sigma"] <= med_sigma]
        if cand.empty:
            cand = rem
        side = cand.sort_values(["abs_mu", "sigma"], ascending=[True, True]).index[0]
        labels[side] = "Sideways"
        remaining.discard(side)

    # 5) Steady Growth: positive μ (not Bull/Goldilocks), moderate σ
    if remaining:
        rem = stats.loc[list(remaining)]
        cand = rem[rem["mu"] > 0]
        if not cand.empty:
            # prefer higher Sharpe but not already assigned
            steady = cand.sort_values(["sharpe", "mu", "sigma"], ascending=[False, False, True]).index[0]
            labels[steady] = "Steady Growth"
            remaining.discard(steady)

    # 6) Risk-Off: negative μ, highest σ (if none neg, pick highest σ)
    if remaining:
        rem = stats.loc[list(remaining)]
        cand = rem[rem["mu"] <= 0]
        if cand.empty:
            cand = rem
        risk = cand["sigma"].idxmax()
        labels[risk] = "Risk-Off"
        remaining.discard(risk)

    # Any leftover: fallbacks
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

    # compress consecutive runs
    runs = []
    cur_state = None
    start = None
    prev_dt = None
    for dt, row in df.iterrows():
        st = row["state"]
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


def _format_report(tl: pd.DataFrame) -> str:
    lines = ["REGIME ANALYSIS", "================", ""]
    for _, r in tl.iterrows():
        lines += [
            f"PERIOD {int(r['period_index'])}: {r['label']}",
            f"   Duration: {r['start']} to {r['end']}",
            f"   Length: {int(r['trading_days'])} trading days ({r['years']:.1f} years)",
            f"   Annual Return: {r['ann_return']*100:.1f}%",
            f"   Annual Volatility: {r['ann_vol']*100:.1f}%",
            f"   Sharpe Ratio: {r['sharpe']:.2f}",
            f"   Max Drawdown: {r['max_drawdown']*100:.1f}%",
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
    return "\n".join(lines)


# ---------------- Public entrypoint ----------------
def stable_regime_analysis(
    assets: Any,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    n_components: int | str = "auto",
    # NEW: guardrails for auto-K so we don't end up with just 2 states
    k_floor: int = 4,
    k_cap: int = 6,
    auto_k_metric: str = "bic",
    sticky: float = 0.98,
    min_segment_days: int = 20,
    return_result: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> dict | AnalysisResult | str:
    """
    Run sticky-HMM regime analysis and return report and timeline.

    - If n_components="auto", we choose K in [k_floor, k_cap] by BIC/AIC (default BIC).
    - Rich labeling maps hidden states to: Goldilocks / Bull / Steady Growth / Sideways / Risk-Off / Bear (subset if K < 6).
    """
    # Load prices and compute daily log-returns
    px = _load_prices(assets, start_date, end_date)
    if px.shape[1] != 1:
        px = px.iloc[:, [0]]
    prices = px.iloc[:, 0].dropna().astype(float)
    returns = _compute_returns(prices)

    # Preprocess: winsorize + robust scale
    r_w = _winsorize(returns, p=0.005)
    scaler = RobustScaler()
    X = scaler.fit_transform(r_w.values.reshape(-1, 1)).astype(float)

    # --- Auto-K selection with floor/ceiling ---
    def _ic_for_k(k: int, metric: str = "bic") -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(
                n_components=k,
                covariance_type="full",
                n_iter=200,
                tol=1e-4,
                random_state=random_state,
                algorithm="viterbi",
                init_params="mc",
            )
            # Sticky prior via diag-heavy transmat init
            trans = np.full((k, k), (1 - sticky) / (k - 1))
            np.fill_diagonal(trans, sticky)
            model.startprob_ = np.full(k, 1.0 / k)
            model.transmat_ = trans
            model.fit(X)
            logL = model.score(X)
            # naive parameter count
            p = k + k + k * k
            if metric == "aic":
                return float(-2 * logL + 2 * p)
            # default bic
            return float(-2 * logL + p * np.log(len(X)))

    if isinstance(n_components, str) and n_components == "auto":
        lo = max(2, int(k_floor))
        hi = max(lo, int(k_cap))
        candidate_ks = list(range(lo, hi + 1))
        ics = {k: _ic_for_k(k, auto_k_metric.lower()) for k in candidate_ks}
        K = min(ics, key=ics.get)
    else:
        K = int(n_components)

    # --- Final fit with chosen K ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GaussianHMM(
            n_components=K,
            covariance_type="full",
            n_iter=500,
            tol=1e-4,
            random_state=random_state,
            algorithm="viterbi",
            init_params="mc",
        )
        trans = np.full((K, K), (1 - sticky) / (K - 1))
        np.fill_diagonal(trans, sticky)
        model.startprob_ = np.full(K, 1.0 / K)
        model.transmat_ = trans
        model.fit(X)
        states = model.predict(X)

    # Enforce minimum segment length
    states = _min_segment_enforce(states, min_segment_days)

    # Rich labels (μ, σ, Sharpe)
    labels = _label_states_rich(returns.iloc[: len(states)], states)

    # Build timeline
    tl = _build_timeline(
        returns.index[: len(states)],
        states,
        returns,
        prices.iloc[1 : len(states) + 1],
        labels,
    )

    report = _format_report(tl)
    meta = {
        "method": "hmm_sticky",
        "n_components": K,
        "k_floor": k_floor,
        "k_cap": k_cap,
        "auto_k_metric": auto_k_metric,
        "sticky": sticky,
        "min_segment_days": min_segment_days,
        "random_state": random_state,
        "labeler": "mu_sigma_sharpe_v1",
        "states_present": int(len(set(states.tolist()))),
    }

    result = AnalysisResult(report=report, timeline=tl, meta=meta)

    if return_result:
        regime_timeline = tl.to_dict(orient="records")
        return {"report": report, "regime_timeline": regime_timeline, "meta": meta}
    else:
        return report


# ---------------- Convenience ----------------
def stable_report(*args, **kwargs) -> str:
    """Return only the human-readable report text."""
    kwargs["return_result"] = True
    res = stable_regime_analysis(*args, **kwargs)
    return res["report"]