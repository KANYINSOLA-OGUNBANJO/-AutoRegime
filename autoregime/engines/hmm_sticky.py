"""
Sticky HMM engine for AutoRegime (drop-in)

- Robust preprocessing (log-returns, winsorize, robust scale)
- Sticky transition prior (diag-heavy init) + Viterbi decode
- Minimum segment length enforcement to stop 1-day flip-flops
- Clean timeline DataFrame + human-readable report
- Works with a ticker (via yfinance) or a price Series/DataFrame

Public entrypoint (compatible):
    stable_regime_analysis(assets, start_date=None, end_date=None,
                           n_components='auto', sticky=0.98,
                           min_segment_days=20, return_result=True,
                           random_state=42, verbose=True)

Convenience:
    stable_report(...) -> str   # returns only the formatted report text
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

# Optional dependency: only import if a ticker string is passed
try:  # pragma: no cover - optional runtime dep
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # noqa: N816

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

TRADING_DAYS = 252


def _annualize_return(daily_returns: pd.Series) -> float:
    mu = daily_returns.mean() * TRADING_DAYS
    return float(mu)


def _annualize_vol(daily_returns: pd.Series) -> float:
    sig = daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
    return float(sig)


def _max_drawdown_from_prices(prices: pd.Series) -> float:
    cummax = prices.cummax()
    dd = (prices / cummax - 1.0).min()
    return float(dd)


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


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def _load_prices(assets: Any, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """
    Return a price DataFrame with one column if assets is a single ticker/Series.
    Handles yfinance returning either a Series or a DataFrame for Close.
    """
    # Ticker string
    if isinstance(assets, str):
        if yf is None:
            raise RuntimeError(
                "yfinance is required to download prices for string tickers. Install with: pip install yfinance"
            )
        try:
            df = yf.download(assets, start=start_date, end=end_date, auto_adjust=True, progress=False)
        except Exception as e:  # network/DNS issues etc.
            raise RuntimeError(f"Failed to download data for {assets} via yfinance: {e}") from e

        if df is None or df.empty:
            raise ValueError(f"No data returned for {assets}.")

        # yfinance can return either:
        # - DataFrame with columns ['Open','High','Low','Close','Adj Close','Volume']
        #   where df['Close'] is a Series (usual), OR
        # - DataFrame where df['Close'] itself is a DataFrame (e.g., multi-ticker pattern or provider quirk)
        close = df["Close"]

        if isinstance(close, pd.DataFrame):
            # Reduce to first column and ensure it's a Series
            if close.shape[1] < 1:
                raise ValueError(f"No Close price columns returned for {assets}.")
            ser = close.iloc[:, 0].astype(float)
        else:
            # Series path
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


# ---------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------
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
            # Prefer merging into right if available; else left; else keep
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


def _label_states_by_mean_return(returns: pd.Series, states: np.ndarray) -> dict[int, str]:
    df = pd.DataFrame({"state": states}, index=returns.index).join(returns.rename("r"))
    means = df.groupby("state")["r"].mean().sort_values()
    labels: dict[int, str] = {}
    k = len(means)
    for rank, (state, mu) in enumerate(means.items()):
        if rank == 0:
            labels[state] = "Bear Market"
        elif rank == k - 1:
            labels[state] = "Bull Market"
        else:
            labels[state] = "Steady Growth" if mu > 0 else "Correction"
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


# ---------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------
def stable_regime_analysis(
    assets: Any,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    n_components: int | str = "auto",
    sticky: float = 0.98,
    min_segment_days: int = 20,
    return_result: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> dict | AnalysisResult | str:
    """
    Run sticky-HMM regime analysis and return report and timeline.

    If `return_result=True`, returns a dict with key "regime_timeline"
    to preserve existing callers.
    """
    # Load prices and compute daily log-returns
    px = _load_prices(assets, start_date, end_date)
    if px.shape[1] != 1:
        # For now, reduce to the first column; multivariate can be added later
        px = px.iloc[:, [0]]
    prices = px.iloc[:, 0].dropna().astype(float)
    returns = _compute_returns(prices)

    # Preprocess: winsorize + robust scale to make HMM happy
    r_w = _winsorize(returns, p=0.005)
    scaler = RobustScaler()
    X = scaler.fit_transform(r_w.values.reshape(-1, 1)).astype(float)

    # Choose K automatically (simple BIC sweep) if needed
    def _bic_for_k(k: int) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(
                n_components=k,
                covariance_type="full",
                n_iter=200,
                tol=1e-4,
                random_state=random_state,
                algorithm="viterbi",
                init_params="mc",  # keep our startprob_ and transmat_
            )
            # Sticky prior via diag-heavy transmat init
            trans = np.full((k, k), (1 - sticky) / (k - 1))
            np.fill_diagonal(trans, sticky)
            model.startprob_ = np.full(k, 1.0 / k)
            model.transmat_ = trans
            model.fit(X)
            # BIC = -2 logL + p logN
            logL = model.score(X)
            # parameters ~ means(k) + covars(k) + trans(k*k) naive count
            p = k + k + k * k
            bic = -2 * logL + p * np.log(len(X))
            return float(bic)

    if isinstance(n_components, str) and n_components == "auto":
        candidate_ks = [2, 3, 4, 5, 6]
        bics = {k: _bic_for_k(k) for k in candidate_ks}
        K = min(bics, key=bics.get)
    else:
        K = int(n_components)

    # Final model fit with chosen K
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GaussianHMM(
            n_components=K,
            covariance_type="full",
            n_iter=500,
            tol=1e-4,
            random_state=random_state,
            algorithm="viterbi",
            init_params="mc",  # keep our startprob_ and transmat_
        )
        trans = np.full((K, K), (1 - sticky) / (K - 1))
        np.fill_diagonal(trans, sticky)
        model.startprob_ = np.full(K, 1.0 / K)
        model.transmat_ = trans
        model.fit(X)
        states = model.predict(X)

    # Enforce minimum segment length
    states = _min_segment_enforce(states, min_segment_days)

    # Label states by mean return
    labels = _label_states_by_mean_return(returns.iloc[: len(states)], states)

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
        "sticky": sticky,
        "min_segment_days": min_segment_days,
        "random_state": random_state,
    }

    result = AnalysisResult(report=report, timeline=tl, meta=meta)

    if return_result:
        # Preserve older calling style ar.stable_regime_analysis(...)["regime_timeline"]
        regime_timeline = tl.to_dict(orient="records")
        return {"report": report, "regime_timeline": regime_timeline, "meta": meta}
    else:
        return report


# ---------------------------------------------------------------------
# Convenience helper (ADD BELOW stable_regime_analysis — not inside it)
# ---------------------------------------------------------------------
def stable_report(*args, **kwargs) -> str:
    """
    Convenience wrapper that returns only the human-readable report text.
    Usage:
        import autoregime as ar
        print(ar.stable_report("SPY", start_date="2015-01-01"))
    """
    kwargs["return_result"] = True
    res = stable_regime_analysis(*args, **kwargs)
    return res["report"]