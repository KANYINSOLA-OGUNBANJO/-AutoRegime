# autoregime/engines/hmm_sticky.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import warnings, numpy as np, pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler

from ..reporting.common import (
    TRADING_DAYS, MIN_CAGR_DAYS,
    ensure_positive_prices, compute_log_returns,
    get_daily_risk_free, compute_excess_log_returns,
    winsorize, ensure_finite_series,
    build_timeline_from_state_runs, format_report,
    annotate_intra_regime_shocks,
)

try:
    from ..core.validation import validate_timeline  # type: ignore
    _HAS_VALIDATOR = True
except Exception:
    _HAS_VALIDATOR = False

@dataclass
class AnalysisResult:
    report: str
    timeline: pd.DataFrame
    meta: dict

# ---------- I/O ----------
def _load_prices(assets: Any, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    if isinstance(assets, str):
        if yf is None:
            raise RuntimeError("yfinance is required. pip install yfinance")
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
        return ensure_positive_prices(assets).to_frame(name=name)

    if isinstance(assets, pd.DataFrame):
        if assets.shape[1] < 1:
            raise ValueError("Price DataFrame has no columns.")
        df = assets.astype(float).apply(ensure_positive_prices)
        first = df.columns[0]
        return df[[first]].dropna()

    raise TypeError("assets must be a ticker str, pandas Series, or DataFrame of prices")

# ---------- helpers ----------
def _runs_from_states(states: np.ndarray) -> List[Tuple[int, int, int]]:
    runs: List[Tuple[int, int, int]] = []
    if len(states) == 0:
        return runs
    i = 0
    while i < len(states):
        j = i
        while j + 1 < len(states) and states[j + 1] == states[i]:
            j += 1
        runs.append((i, j, int(states[i])))
        i = j + 1
    return runs

def _min_segment_enforce(states: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1 or len(states) == 0:
        return states
    s = states.copy().astype(int)
    changed = True
    while changed:
        changed = False
        runs = _runs_from_states(s)
        for idx, (a, b, _) in enumerate(runs):
            if (b - a + 1) >= min_len:
                continue
            left_state  = runs[idx-1][2] if idx-1 >= 0 else None
            right_state = runs[idx+1][2] if idx+1 < len(runs) else None
            left_len  = (runs[idx-1][1]-runs[idx-1][0]+1) if left_state  is not None else -1
            right_len = (runs[idx+1][1]-runs[idx+1][0]+1) if right_state is not None else -1
            if right_len >= left_len and right_state is not None:
                s[a:b+1] = right_state
            elif left_state is not None:
                s[a:b+1] = left_state
            else:
                continue
            changed = True
            break
    return s

def _label_states_rich(returns: pd.Series, states: np.ndarray) -> dict[int, str]:
    df = pd.DataFrame({"state": states}, index=returns.index).join(returns.rename("r"))
    g = df.groupby("state")["r"]
    stats = pd.DataFrame({"mu": g.mean(), "sigma": g.std(ddof=0).replace(0, np.nan)})
    stats["sharpe"] = stats["mu"] / stats["sigma"]

    labels: dict[int, str] = {}
    remaining = set(stats.index)
    pos_idx = stats.index[stats["mu"] > 0].tolist()
    neg_idx = stats.index[stats["mu"] < 0].tolist()

    if len(pos_idx) > 0:
        pos_stats = stats.loc[pos_idx].sort_values("sharpe", ascending=False)
        if pos_stats["sharpe"].notna().any():
            gld = int(pos_stats.index[0]); labels[gld] = "Goldilocks"; remaining.discard(gld)

    if len(neg_idx) > 0:
        br = int(stats.loc[neg_idx, "mu"].idxmin()); labels[br] = "Bear Market"; remaining.discard(br)

    rem_pos = [i for i in remaining if stats.loc[i, "mu"] > 0]
    if len(rem_pos) > 0:
        bull = int(stats.loc[rem_pos, "mu"].idxmax()); labels[bull] = "Bull Market"; remaining.discard(bull)

    if len(remaining) > 0:
        rem = stats.loc[list(remaining)].copy(); rem["abs_mu"] = rem["mu"].abs()
        med_sigma = float(stats["sigma"].median())
        cand = rem[rem["sigma"] <= med_sigma] if np.isfinite(med_sigma) else rem
        side = int(cand.sort_values(["abs_mu","sigma"], ascending=[True, True]).index[0])
        labels[side] = "Sideways"; remaining.discard(side)

    for st in list(remaining):
        labels[int(st)] = "Steady Growth" if stats.loc[st, "mu"] > 0 else "Risk-Off"
    return labels

def _relabel_from_segment_metrics(tl: pd.DataFrame) -> pd.DataFrame:
    if tl is None or tl.empty:
        return tl.copy()
    out = tl.copy()
    vol_col = "ann_vol" if "ann_vol" in out.columns else None
    if vol_col and out[vol_col].notna().any():
        vol_med = float(np.nanmedian(out[vol_col])); vol_p70 = float(np.nanpercentile(out[vol_col].dropna(), 70))
    else:
        vol_med = np.nan; vol_p70 = np.nan

    for i, r in out.iterrows():
        mu_ann = float(r.get("ann_return_mean", np.nan))
        vol    = float(r.get("ann_vol", np.nan))
        mdd    = float(r.get("max_drawdown", np.nan))
        ret_tot= float(r.get("period_return", np.nan))
        sharpe = float(r.get("sharpe", np.nan))
        old    = str(r.get("label", ""))

        if not (np.isfinite(mu_ann) and np.isfinite(vol) and np.isfinite(ret_tot) and np.isfinite(sharpe)):
            continue

        NEAR_ZERO_MU = 0.02; SMALL_MOVE = 0.10
        LOW_VOL  = (np.isfinite(vol_med) and vol <= 1.20 * vol_med)
        HIGH_VOL = (np.isfinite(vol_p70) and vol >= vol_p70)
        new_lab = old

        if mu_ann <= -0.03 and (sharpe <= -0.30 or mdd <= -0.25):
            new_lab = "Bear Market"
        elif mu_ann < -0.01 and HIGH_VOL:
            new_lab = "Risk-Off"
        elif abs(mu_ann) < NEAR_ZERO_MU and abs(ret_tot) < SMALL_MOVE and LOW_VOL:
            new_lab = "Sideways"
        elif mu_ann > 0.03 and sharpe >= 1.50 and (LOW_VOL or mdd >= -0.15):
            new_lab = "Goldilocks"
        elif mu_ann > 0.02 and sharpe >= 0.70:
            new_lab = "Bull Market"
        elif mu_ann > 0.0:
            new_lab = "Steady Growth"
        elif mu_ann < 0.0:
            new_lab = "Correction"

        if new_lab != old:
            out.at[i, "label"] = new_lab
            prev = str(out.at[i, "note"] or "").strip()
            out.at[i, "note"] = (prev + (", " if prev else "") + "segment_relabel_v3")
    return out

# ---------- public entrypoint ----------
def stable_regime_analysis(
    assets: Any,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    n_components: int | str = "auto",
    k_floor: int = 4,
    k_cap: int = 6,
    auto_k_metric: str = "bic",
    sticky: float = 0.98,
    min_segment_days: int = 20,
    show_cagr: bool = False,
    return_result: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> dict | AnalysisResult | str:

    # Load prices (strictly positive) and returns
    px = _load_prices(assets, start_date, end_date)
    if px.shape[1] != 1:
        px = px.iloc[:, [0]]
    prices = px.iloc[:, 0].dropna().astype(float)
    prices = ensure_positive_prices(prices)

    data_earliest = str(pd.Timestamp(prices.index.min()).date()) if len(prices) else ""
    data_latest   = str(pd.Timestamp(prices.index.max()).date()) if len(prices) else ""

    r = compute_log_returns(prices)
    if len(r) < max(60, min_segment_days * 3):
        report = "REGIME ANALYSIS\n================\n\nInsufficient data to segment reliably."
        meta = {
            "method": "hmm_sticky",
            "n_components": 0, "sticky": sticky, "min_segment_days": min_segment_days,
            "random_state": random_state, "states_present": 0, "n_obs": int(len(r)),
            "data_range": {"earliest": data_earliest, "latest": data_latest},
            "notes": {"cagr_suppressed_below_days": MIN_CAGR_DAYS, "show_cagr": bool(show_cagr)},
            "validation": {"ok": True, "errors": [], "warnings": ["short_series"], "info": {}},
        }
        return {"report": report, "regime_timeline": [], "meta": meta} if return_result else report

    # Risk-free → excess returns for Sharpe; segmentation uses winsorized excess
    rf_daily = get_daily_risk_free(r.index.min(), r.index.max(), index=r.index, series="GS10", mode="cc")
    rx = compute_excess_log_returns(r, rf_daily)             # finite by construction
    r_w = winsorize(rx, p=0.005)
    r_w = ensure_finite_series(r_w, fill=np.nan).dropna()
    if r_w.nunique() <= 1:
        # Avoid zero-IQR issues for RobustScaler by adding tiny jitter
        rs = np.random.RandomState(random_state)
        r_w = r_w + (1e-8 * rs.normal(size=len(r_w)))

    # Design matrix (N, 1) with strict finiteness
    X = r_w.values.reshape(-1, 1).astype(float)
    if not np.isfinite(X).all():
        mask = np.isfinite(X[:, 0])
        X = X[mask]
        r_w = r_w.iloc[mask]
    if len(X) < max(60, min_segment_days * 3):
        report = "REGIME ANALYSIS\n================\n\nInsufficient clean data after sanitization."
        return {"report": report, "regime_timeline": [], "meta": {"n_obs": int(len(X))}}

    scaler = RobustScaler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xs = scaler.fit_transform(X)

    # --- Auto-K selection ---
    def _ic_for_k(k: int, metric: str = "bic") -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(
                n_components=k, covariance_type="full", n_iter=200, tol=1e-4,
                random_state=random_state, algorithm="viterbi", init_params="mc"
            )
            trans = np.full((k, k), (1 - sticky) / (k - 1)); np.fill_diagonal(trans, sticky)
            model.startprob_ = np.full(k, 1.0 / k); model.transmat_ = trans
            model.fit(Xs); logL = model.score(Xs)
            p = k + k + k * k
            return float(-2 * logL + (2 * p if metric.lower()=="aic" else p * np.log(len(Xs))))

    if isinstance(n_components, str) and n_components == "auto":
        lo = max(2, int(k_floor))
        hi = min(max(lo, int(k_cap)), max(lo, max(2, len(Xs) // 20)))
        ics = {k: _ic_for_k(k, auto_k_metric) for k in range(lo, hi + 1)}
        K = min(ics, key=ics.get)
    else:
        K = int(n_components)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GaussianHMM(
            n_components=K, covariance_type="full", n_iter=500, tol=1e-4,
            random_state=random_state, algorithm="viterbi", init_params="mc"
        )
        trans = np.full((K, K), (1 - sticky) / (K - 1)); np.fill_diagonal(trans, sticky)
        model.startprob_ = np.full(K, 1.0 / K); model.transmat_ = trans
        model.fit(Xs)
        states = model.predict(Xs)

    states = _min_segment_enforce(states, min_segment_days)

    labels = _label_states_rich(r_w, states)

    # Align prices to r_w index (same dates — simpler & robust)
    px_aligned = prices.reindex(r_w.index).dropna()
    # Build timeline + annotate shocks
    tl = build_timeline_from_state_runs(
        index=r_w.index, states=states, returns=r_w, prices_aligned_to_returns=px_aligned, state_to_label=labels
    )
    tl = _relabel_from_segment_metrics(tl)
    tl = annotate_intra_regime_shocks(tl, returns=r.loc[r_w.index])

    # Validate
    if _HAS_VALIDATOR:
        try:
            v = validate_timeline(tl, min_segment_days=min_segment_days, min_cagr_days=MIN_CAGR_DAYS)
            validation_meta = {"ok": v.ok, "errors": v.errors, "warnings": v.warnings, "info": v.info}
        except Exception as _vex:
            validation_meta = {"ok": False, "errors": [f"validator exception: {_vex}"], "warnings": [], "info": {}}
    else:
        validation_meta = {"ok": True, "errors": [], "warnings": ["validator_not_installed"], "info": {}}

    report = format_report(tl, show_cagr=show_cagr, hide_cagr_line=True, title="REGIME ANALYSIS")
    meta = {
        "method": "hmm_sticky",
        "n_components": K, "k_floor": k_floor, "k_cap": k_cap, "auto_k_metric": auto_k_metric,
        "sticky": sticky, "min_segment_days": min_segment_days, "random_state": random_state,
        "labeler": "mu_sigma_sharpe_v2 + segment_relabel_v3",
        "states_present": int(len(set(states.tolist()))), "n_obs": int(len(r_w)),
        "data_range": {"earliest": data_earliest, "latest": data_latest},
        "notes": {"cagr_suppressed_below_days": MIN_CAGR_DAYS, "show_cagr": bool(show_cagr)},
        "validation": validation_meta,
    }

    if return_result:
        return {"report": report, "regime_timeline": tl.to_dict(orient="records"), "meta": meta}
    else:
        return report

def stable_report(*args, **kwargs) -> str:
    kwargs.setdefault("return_result", True)
    kwargs.setdefault("show_cagr", False)
    res = stable_regime_analysis(*args, **kwargs)
    return res["report"]