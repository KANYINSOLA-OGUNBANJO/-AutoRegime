# autoregime/engines/hmm_sticky.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd

# yfinance is optional (the app often passes a Series already)
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler

from ..reporting.common import (
    TRADING_DAYS,
    MIN_CAGR_DAYS,
    ensure_positive_prices,
    compute_log_returns,
    get_daily_risk_free,
    compute_excess_log_returns,
    winsorize,
    ensure_finite_series,
    build_timeline_from_state_runs,
    format_report,
    annotate_intra_regime_shocks,
)

# Optional validator
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


# --------------------------------------------------------------------------------------
# Price loader: accepts a ticker string, Series, or DataFrame; returns a 1-col DataFrame
# --------------------------------------------------------------------------------------
def _load_prices(assets: Any, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    # Ticker symbol (string): fetch with yfinance
    if isinstance(assets, str):
        if yf is None:
            raise RuntimeError("yfinance is required to load a ticker string. Try passing a Series instead.")
        # yfinance `end` is exclusive; allow None
        df = yf.download(assets, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if df is None or df.empty or "Close" not in df.columns:
            raise ValueError(f"No data returned for {assets}.")
        ser = df["Close"]
        if isinstance(ser, pd.DataFrame):  # single-column edge case
            ser = ser.iloc[:, 0]
        ser = ensure_positive_prices(ser.astype(float))
        ser.name = str(assets)
        return ser.to_frame()

    # Already a Series of prices
    if isinstance(assets, pd.Series):
        name = getattr(assets, "name", None) or "asset"
        ser = ensure_positive_prices(pd.to_numeric(assets, errors="coerce").astype(float))
        return ser.to_frame(name=name)

    # DataFrame: take first column as prices
    if isinstance(assets, pd.DataFrame):
        if assets.shape[1] < 1:
            raise ValueError("Price DataFrame has no columns.")
        df = assets.copy()
        first = df.columns[0]
        ser = ensure_positive_prices(pd.to_numeric(df[first], errors="coerce").astype(float))
        ser.name = str(first)
        return ser.to_frame()

    raise TypeError("assets must be a ticker str, pandas Series, or DataFrame of prices")


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _runs_from_states(states: np.ndarray) -> List[Tuple[int, int, int]]:
    runs: List[Tuple[int, int, int]] = []
    if len(states) == 0:
        return runs
    i, n = 0, len(states)
    while i < n:
        j = i
        while j + 1 < n and states[j + 1] == states[i]:
            j += 1
        runs.append((i, j, int(states[i])))
        i = j + 1
    return runs


def _min_segment_enforce(states: np.ndarray, min_len: int) -> np.ndarray:
    """Merge sub-min segments into the larger neighbor (iteratively)."""
    if min_len <= 1 or len(states) == 0:
        return states
    s = states.copy().astype(int)
    changed = True
    while changed:
        changed = False
        runs = _runs_from_states(s)
        for idx, (a, b, _) in enumerate(runs):
            seg_len = b - a + 1
            if seg_len >= min_len:
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
    Label by state-level stats (mu, sigma, Sharpe) on the provided returns Series.
    NOTE: pass **excess returns** here if you want Sharpe-like ordering.
    """
    df = pd.DataFrame({"state": states}, index=returns.index).join(returns.rename("r"))
    g = df.groupby("state")["r"]
    stats = pd.DataFrame({"mu": g.mean(), "sigma": g.std(ddof=0).replace(0, np.nan)})
    stats["sharpe"] = stats["mu"] / stats["sigma"]

    labels: dict[int, str] = {}
    remaining = set(stats.index)
    pos_idx = stats.index[stats["mu"] > 0].tolist()
    neg_idx = stats.index[stats["mu"] < 0].tolist()

    # Goldilocks: best risk-adjusted among positive states
    if len(pos_idx) > 0:
        pos_stats = stats.loc[pos_idx].sort_values("sharpe", ascending=False)
        if pos_stats["sharpe"].notna().any():
            gld = int(pos_stats.index[0])
            labels[gld] = "Goldilocks"
            remaining.discard(gld)

    # Bear: worst mean
    if len(neg_idx) > 0:
        br = int(stats.loc[neg_idx, "mu"].idxmin())
        labels[br] = "Bear Market"
        remaining.discard(br)

    # Bull: remaining positive with highest mean
    rem_pos = [i for i in remaining if stats.loc[i, "mu"] > 0]
    if len(rem_pos) > 0:
        bull = int(stats.loc[rem_pos, "mu"].idxmax())
        labels[bull] = "Bull Market"
        remaining.discard(bull)

    # Sideways: lowest |mu| among low-vol (<= median sigma)
    if len(remaining) > 0:
        rem = stats.loc[list(remaining)].copy()
        rem["abs_mu"] = rem["mu"].abs()
        med_sigma = float(stats["sigma"].median())
        cand = rem[rem["sigma"] <= med_sigma] if np.isfinite(med_sigma) else rem
        side = int(cand.sort_values(["abs_mu", "sigma"], ascending=[True, True]).index[0])
        labels[side] = "Sideways"
        remaining.discard(side)

    # The rest:
    for st in list(remaining):
        labels[int(st)] = "Steady Growth" if stats.loc[st, "mu"] > 0 else "Risk-Off"
    return labels


def _relabel_from_segment_metrics(tl: pd.DataFrame) -> pd.DataFrame:
    """
    Segment-level relabeling guardrails so we don't call deep drawdowns 'Goldilocks', etc.
    """
    if tl is None or tl.empty:
        return tl.copy()

    out = tl.copy()
    vol_col = "ann_vol" if "ann_vol" in out.columns else None
    if vol_col and out[vol_col].notna().any():
        vol_med = float(np.nanmedian(out[vol_col]))
        vol_p70 = float(np.nanpercentile(out[vol_col].dropna(), 70))
    else:
        vol_med = np.nan
        vol_p70 = np.nan

    for i, r in out.iterrows():
        mu_ann = float(r.get("ann_return_mean", np.nan))  # annualized mean (excess) inside builder
        vol = float(r.get("ann_vol", np.nan))
        mdd = float(r.get("max_drawdown", np.nan))
        ret_tot = float(r.get("period_return", np.nan))
        sharpe = float(r.get("sharpe", np.nan))
        old = str(r.get("label", ""))

        if not (np.isfinite(mu_ann) and np.isfinite(vol) and np.isfinite(ret_tot) and np.isfinite(sharpe)):
            continue

        NEAR_ZERO_MU = 0.02
        SMALL_MOVE = 0.10
        LOW_VOL = (np.isfinite(vol_med) and vol <= 1.20 * vol_med)
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


# --------------------------------------------------------------------------------------
# HMM utilities
# --------------------------------------------------------------------------------------
def _n_params_gaussian_hmm(k: int, d: int = 1, covariance_type: str = "full") -> int:
    """Rough parameter count for BIC/AIC."""
    # startprob: K-1 free; trans: K*(K-1); means: K*d; covs: K * d(d+1)/2 (full) or K*d (diag)
    start = k - 1
    trans = k * (k - 1)
    means = k * d
    if covariance_type == "full":
        covs = k * (d * (d + 1) // 2)
    else:
        covs = k * d
    return start + trans + means + covs


def _fit_hmm(
    Xs: np.ndarray,
    K: int,
    sticky: float,
    random_state: int,
    n_init: int = 3,
) -> GaussianHMM:
    """
    Fit a sticky GaussianHMM robustly. Ensures finite inputs, small min_covar, and
    diagonal-heavy transmat prior to reduce flip-flops.
    """
    if not np.isfinite(Xs).all():
        mask = np.isfinite(Xs[:, 0])
        Xs = Xs[mask]
    if len(Xs) < 10:
        raise ValueError("Too few observations after sanitization.")

    # sticky Dirichlet prior: each row favors staying in the same state
    off = max(1e-6, (1.0 - float(sticky)) / max(1, K - 1))
    diag = float(sticky)
    trans_prior = np.full((K, K), off, dtype=float)
    np.fill_diagonal(trans_prior, diag)

    # start prob prior: near-uniform (avoid degeneracy)
    start_prior = np.full(K, 1.0 / K, dtype=float)

    best_model = None
    best_score = -np.inf

    # Suppress common hmmlearn warnings that aren't fatal
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        for seed in [random_state + i for i in range(n_init)]:
            model = GaussianHMM(
                n_components=K,
                covariance_type="full",
                min_covar=1e-6,
                n_iter=250,
                tol=1e-4,
                random_state=seed,
                params="stmc",          # (startprob, transmat, means, covars)
                init_params="mc",       # we set priors for s,t; let means/covars init
            )
            # Priors
            model.startprob_prior = start_prior
            model.transmat_prior = trans_prior
            try:
                model.fit(Xs)
                score = float(model.score(Xs))  # log-likelihood
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError("HMM fit failed for all initializations.")
    return best_model


def _choose_k_by_ic(
    Xs: np.ndarray,
    sticky: float,
    random_state: int,
    k_floor: int,
    k_cap: int,
    metric: str = "bic",
) -> Tuple[int, GaussianHMM, dict]:
    """
    Try K in [k_floor..k_cap] (clamped by sample size) and pick best by BIC/AIC.
    Returns (K*, model, scores).
    """
    N = len(Xs)
    # ensure at least ~25 points per state
    k_max_by_n = max(2, min(k_cap, N // 25))
    k_min = max(2, min(k_floor, k_max_by_n))
    k_max = max(k_min, k_max_by_n)

    scores: dict[int, dict] = {}
    best = None
    best_ic = np.inf

    for K in range(k_min, k_max + 1):
        try:
            mdl = _fit_hmm(Xs, K=K, sticky=sticky, random_state=random_state)
            ll = float(mdl.score(Xs))
            p = _n_params_gaussian_hmm(K, d=Xs.shape[1], covariance_type="full")
            if metric.lower() == "aic":
                ic = -2.0 * ll + 2.0 * p
            else:  # BIC default
                ic = -2.0 * ll + p * np.log(max(1, N))
            scores[K] = {"ll": ll, "p": p, "ic": ic}
            if ic < best_ic:
                best_ic = ic
                best = (K, mdl)
        except Exception:
            continue

    if best is None:
        # Final fallback: force K=2
        mdl = _fit_hmm(Xs, K=2, sticky=sticky, random_state=random_state)
        return 2, mdl, {2: {"ll": float(mdl.score(Xs)), "p": _n_params_gaussian_hmm(2), "ic": 0.0}}

    return best[0], best[1], scores


# --------------------------------------------------------------------------------------
# Main entrypoint
# --------------------------------------------------------------------------------------
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
    verbose: bool = False,
) -> dict | AnalysisResult | str:
    """
    Sticky Gaussian HMM over **winsorized raw daily log-returns** (NOT already excess),
    so that Sharpe in the timeline is computed correctly on excess inside the timeline builder.
    """

    # ---- Load & sanitize prices
    px = _load_prices(assets, start_date, end_date)
    if px.shape[1] != 1:
        px = px.iloc[:, [0]]
    prices = ensure_positive_prices(px.iloc[:, 0].dropna().astype(float))

    if len(prices) < 2:
        report = "REGIME ANALYSIS\n================\n\nNot enough price points."
        meta = {"method": "hmm_sticky", "n_obs": int(len(prices))}
        return {"report": report, "regime_timeline": [], "meta": meta} if return_result else report

    data_earliest = str(pd.Timestamp(prices.index.min()).date())
    data_latest = str(pd.Timestamp(prices.index.max()).date())

    # ---- Raw log returns for segmentation features
    r_raw = compute_log_returns(prices)  # index ⊂ prices.index
    r_raw = ensure_finite_series(r_raw, fill=np.nan).dropna()

    if len(r_raw) < max(60, min_segment_days * 3):
        report = "REGIME ANALYSIS\n================\n\nInsufficient data to segment reliably."
        meta = {
            "method": "hmm_sticky",
            "n_components": 0,
            "sticky": sticky,
            "min_segment_days": min_segment_days,
            "random_state": random_state,
            "states_present": 0,
            "n_obs": int(len(r_raw)),
            "data_range": {"earliest": data_earliest, "latest": data_latest},
            "notes": {"cagr_suppressed_below_days": MIN_CAGR_DAYS, "show_cagr": bool(show_cagr)},
            "validation": {"ok": True, "errors": [], "warnings": ["short_series"], "info": {}},
        }
        return {"report": report, "regime_timeline": [], "meta": meta} if return_result else report

    # Winsorize raw returns (robust to outliers); keep **raw** for timeline builder
    r_w_raw = winsorize(r_raw, p=0.005)
    r_w_raw = ensure_finite_series(r_w_raw, fill=np.nan).dropna()

    # prevent degenerate variance
    if r_w_raw.nunique() <= 1:
        rs = np.random.RandomState(random_state)
        r_w_raw = r_w_raw + (1e-8 * rs.normal(size=len(r_w_raw)))

    # ---- Design matrix for HMM from winsorized **raw** returns
    X = r_w_raw.to_numpy(dtype=float).reshape(-1, 1)
    # Hard mask any remaining non-finite
    mask = np.isfinite(X[:, 0])
    if not mask.all():
        X = X[mask]
        r_w_raw = r_w_raw.iloc[mask]

    # Robust scaling keeps distributions stable for HMM
    scaler = RobustScaler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xs = scaler.fit_transform(X)

    # Guard again
    if len(Xs) < max(60, min_segment_days * 3):
        report = "REGIME ANALYSIS\n================\n\nInsufficient clean data after sanitization."
        meta = {"method": "hmm_sticky", "n_obs": int(len(Xs))}
        return {"report": report, "regime_timeline": [], "meta": meta} if return_result else report

    # ---- Choose K and fit
    if isinstance(n_components, str) and n_components.lower() == "auto":
        K, model, ic_scores = _choose_k_by_ic(
            Xs, sticky=sticky, random_state=random_state,
            k_floor=int(k_floor), k_cap=int(k_cap), metric=str(auto_k_metric)
        )
    else:
        K = max(2, int(n_components))
        model = _fit_hmm(Xs, K=K, sticky=sticky, random_state=random_state)
        ic_scores = {K: {"ll": float(model.score(Xs)), "p": _n_params_gaussian_hmm(K), "ic": 0.0}}

    # ---- Predict states
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = model.predict(Xs).astype(int)

    # Enforce minimum segment length
    z2 = _min_segment_enforce(z, min_segment_days)

    # ---- For **labeling** only, compute excess series (to rank states by Sharpe-like metric)
    try:
        rf_daily = get_daily_risk_free(r_w_raw.index.min(), r_w_raw.index.max(), index=r_w_raw.index, series="GS10", mode="cc")
    except Exception:
        rf_daily = pd.Series(0.0, index=r_w_raw.index, name="rf_daily", dtype=float)
    rx_for_labels = compute_excess_log_returns(r_w_raw, rf_daily.reindex(r_w_raw.index).fillna(0.0))

    labels = _label_states_rich(rx_for_labels if len(rx_for_labels) == len(z2) else r_w_raw, z2)

    # ---- Align prices to returns index (these should already match after diff())
    # Use exact same index; no dropping to keep lengths aligned with states
    prices_aligned = prices.reindex(r_w_raw.index)
    # If any NaNs slipped in, forward-fill then back-fill (no gaps expected, but be safe)
    if prices_aligned.isna().any():
        prices_aligned = prices_aligned.ffill().bfill()

    # ---- Build standardized timeline (Sharpe computed on excess inside builder)
    tl = build_timeline_from_state_runs(
        index=r_w_raw.index,
        states=z2,
        returns=r_w_raw,                      # RAW winsorized returns; builder computes excess for Sharpe
        prices_aligned_to_returns=prices_aligned,
        state_to_label=labels,
    )

    # Intra-regime shock notes (uses *raw* r_raw so dates match)
    tl = annotate_intra_regime_shocks(tl, returns=r_raw.reindex(r_w_raw.index))

    # Segment-level relabel guardrails
    tl = _relabel_from_segment_metrics(tl)

    # ---- Validate (if validator present)
    if _HAS_VALIDATOR:
        try:
            v = validate_timeline(tl, min_segment_days=min_segment_days, min_cagr_days=MIN_CAGR_DAYS)
            validation_meta = {"ok": v.ok, "errors": v.errors, "warnings": v.warnings, "info": v.info}
        except Exception as ex:
            validation_meta = {"ok": False, "errors": [f"validator exception: {ex}"], "warnings": [], "info": {}}
    else:
        validation_meta = {"ok": True, "errors": [], "warnings": ["validator_not_installed"], "info": {}}

    # ---- Report
    report = format_report(tl, show_cagr=show_cagr, hide_cagr_line=True, title="REGIME ANALYSIS")

    meta: Dict[str, Any] = {
        "method": "hmm_sticky",
        "n_components": int(K),
        "auto_k_metric": str(auto_k_metric),
        "ic_scores": ic_scores,
        "sticky": float(sticky),
        "min_segment_days": int(min_segment_days),
        "random_state": int(random_state),
        "states_present": int(len(set(z2.tolist()))) if len(z2) else 0,
        "n_obs": int(len(r_w_raw)),
        "data_range": {"earliest": data_earliest, "latest": data_latest},
        "notes": {"cagr_suppressed_below_days": MIN_CAGR_DAYS, "show_cagr": bool(show_cagr)},
        "validation": validation_meta,
    }

    if return_result:
        return {"report": report, "regime_timeline": tl.to_dict(orient="records"), "meta": meta}
    else:
        return report