"""
Bayesian Online Changepoint Detection (BOCPD) engine (lightweight)

- Constant hazard (H) with Student-t predictive (Normal-Inverse-Gamma conjugacy)
- Online posterior over run-length with pruning for speed
- Convert change alerts → segments → professional timeline/report
- Stability controls: min_segment_days (post-merge), cp_threshold (alert threshold)

NOTE: This is a compact, dependency-light BOCPD suitable for daily financial use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.preprocessing import RobustScaler

# Reuse robust I/O + preprocessing from HMM file
from .hmm_sticky import (
    _load_prices,
    _winsorize,
    _compute_returns,
)

TRADING_DAYS = 252


# ---------- small stat helpers ----------
def _ann_return(daily_returns: pd.Series) -> float:
    return float(daily_returns.mean() * TRADING_DAYS)


def _ann_vol(daily_returns: pd.Series) -> float:
    return float(daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS))


def _max_dd_from_prices(prices: pd.Series) -> float:
    cummax = prices.cummax()
    return float((prices / cummax - 1.0).min())


def _student_t_logpdf(x: float, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Predictive Student-t (conjugate Normal-Inverse-Gamma).
    df = 2*alpha
    scale^2 = beta*(kappa+1)/(alpha*kappa)
    """
    nu = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    return (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * (np.log(nu) + np.log(np.pi) + np.log(scale2))
        - ((nu + 1.0) / 2.0) * np.log(1.0 + ((x - mu) ** 2) / (nu * scale2))
    )


def _update_posterior(
    mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray, x: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One-step conjugate update for all run-lengths (vectorized)."""
    kappa_n = kappa + 1.0
    mu_n = (kappa * mu + x) / kappa_n
    alpha_n = alpha + 0.5
    beta_n = beta + (kappa * (x - mu) ** 2) / (2.0 * kappa_n)
    return mu_n, kappa_n, alpha_n, beta_n


# ---------- core BOCPD ----------
def _bocpd_run(
    x: np.ndarray,
    hazard: float = 1.0 / 120.0,
    cp_threshold: float = 0.35,
    max_run: int = 1000,
    prune: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Online posterior recursion (Adams & MacKay 2007).
    Returns:
      - cp_prob: P(changepoint at t) array, shape (T,)
      - rl_map:  MAP run-length at each t (argmax posterior), shape (T,)
    """
    T = len(x)
    cp_prob = np.zeros(T)
    rl_map = np.zeros(T, dtype=int)

    # vague prior
    mu0, kappa0, alpha0, beta0 = 0.0, 1e-6, 1e-6, 1e-6

    # start with run-length support {0}
    weights = np.array([1.0])
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])

    for t in range(T):
        xt = float(x[t])

        # predictive likelihood for each current run-length
        logpred = _student_t_logpdf(xt, mu, kappa, alpha, beta)
        pred = np.exp(logpred - np.max(logpred))  # stabilize
        s = pred.sum()
        pred = pred / s if s > 0 else pred

        # growth and changepoint paths
        growth = (1.0 - hazard) * pred * weights
        cp = hazard * float(np.sum(pred * weights))

        new_weights = np.empty(growth.shape[0] + 1)
        new_weights[0] = cp  # r_t = 0 (changepoint)
        new_weights[1:] = growth  # r_t = r_{t-1} + 1

        # normalize
        Z = new_weights.sum()
        if Z > 0:
            new_weights /= Z

        cp_prob[t] = new_weights[0]
        rl_map[t] = int(np.argmax(new_weights))

        # update sufficient stats
        mu0n, kappa0n, alpha0n, beta0n = _update_posterior(
            np.array([mu0]), np.array([kappa0]), np.array([alpha0]), np.array([beta0]), xt
        )
        mu_g, kappa_g, alpha_g, beta_g = _update_posterior(mu, kappa, alpha, beta, xt)

        mu = np.concatenate([mu0n, mu_g])
        kappa = np.concatenate([kappa0n, kappa_g])
        alpha = np.concatenate([alpha0n, alpha_g])
        beta = np.concatenate([beta0n, beta_g])
        weights = new_weights

        # prune to keep things fast
        keep = weights > prune
        if keep.sum() == 0:
            keep = np.array([True] + [False] * (len(weights) - 1))
        if keep.sum() > max_run:
            top = np.argsort(weights)[-max_run:]
            keep = np.zeros_like(weights, dtype=bool)
            keep[top] = True

        mu, kappa, alpha, beta, weights = mu[keep], kappa[keep], alpha[keep], beta[keep], weights[keep]

    return cp_prob, rl_map


def _segments_from_cp(cp_prob: np.ndarray, index: pd.DatetimeIndex, min_len: int, threshold: float) -> List[Tuple[int, int]]:
    """
    Turn cp posteriors into (start_idx, end_idx) inclusive segments over the returns index.
    """
    T = len(index)
    if T == 0:
        return []

    segs: List[Tuple[int, int]] = []
    s = 0
    for t in range(T):
        if t > s and cp_prob[t] >= threshold and (t - s) >= min_len:
            segs.append((s, t - 1))
            s = t
    if s <= T - 1:
        segs.append((s, T - 1))

    # enforce min_len greedily (merge short tails)
    merged: List[Tuple[int, int]] = []
    for (a, b) in segs:
        if not merged:
            merged.append((a, b))
        else:
            (pa, pb) = merged[-1]
            if (b - a + 1) < min_len:
                merged[-1] = (pa, b)
            else:
                merged.append((a, b))
    return merged


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
    hazard: float = 1.0 / 120.0,         # ~ one break per ~120 obs on average
    cp_threshold: float = 0.35,          # alert threshold for r_t = 0 posterior
    min_segment_days: int = 20,          # merge very short segments
    return_result: bool = True,
    verbose: bool = True,
) -> dict | AnalysisResult | str:
    """
    BOCPD-based regime analysis with unified output.
    """
    # 1) Load prices & daily log-returns
    px = _load_prices(assets, start_date, end_date)
    prices = px.iloc[:, 0].dropna().astype(float)
    r = _compute_returns(prices)

    # 2) Robustify for the detector (winsorize + robust scale)
    r_w = _winsorize(r, p=0.005)
    scaler = RobustScaler()
    X = scaler.fit_transform(r_w.values.reshape(-1, 1)).astype(float).ravel()

    # 3) Online BOCPD
    cp_prob, _ = _bocpd_run(X, hazard=hazard, cp_threshold=cp_threshold)

    # 4) Convert to segments on the *returns* index
    segs = _segments_from_cp(cp_prob, r.index, min_len=min_segment_days, threshold=cp_threshold)
    if len(segs) == 0:
        segs = [(0, len(r) - 1)]

    # 5) Build timeline with standard columns
    rows = []
    for i, (a, b) in enumerate(segs, start=1):
        r_seg = r.iloc[a:b + 1]
        p_seg = prices.iloc[a + 1:b + 2]  # align with returns window
        mu = _ann_return(r_seg)
        sig = _ann_vol(r_seg)
        sharpe = float(mu / sig) if sig > 0 else np.nan
        mdd = _max_dd_from_prices(p_seg)
        label = "Bull Market" if mu > 0 else "Bear Market"
        rows.append({
            "period_index": i,
            "label": label,
            "state": 1 if label.startswith("Bull") else 0,
            "start": r_seg.index[0].date().isoformat(),
            "end": r_seg.index[-1].date().isoformat(),
            "trading_days": int(len(r_seg)),
            "years": float(len(r_seg) / TRADING_DAYS),
            "ann_return": float(mu),
            "ann_vol": float(sig),
            "sharpe": float(sharpe),
            "max_drawdown": float(mdd),
        })
    tl = pd.DataFrame(rows)

    # 6) Pretty text report
    lines = ["REGIME ANALYSIS", "================", ""]
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
        "method": "bocpd",
        "hazard": hazard,
        "cp_threshold": cp_threshold,
        "min_segment_days": min_segment_days,
    }

    if return_result:
        return {"report": report, "regime_timeline": tl.to_dict(orient="records"), "meta": meta}
    return report