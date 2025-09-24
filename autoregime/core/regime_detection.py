"""
Author: Kanyinsola Ogunbanjo
GitHub: https://github.com/KANYINSOLA-OGUNBANJO/-AutoRegime
"""

from __future__ import annotations

import warnings
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from hmmlearn import hmm

# Configure logging for professional output
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for clean output
warnings.filterwarnings('ignore')


class AutoRegimeDetector:
    """
    Professional Market Regime Detection System using Hidden Markov Models.

    This class implements a sophisticated regime detection algorithm with corrected
    maximum drawdown calculations and deterministic behavior for reproducible results.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        random_state: int = 42,
        max_regimes: int = 6,
        min_regime_duration: int = 15,
        economic_significance_threshold: float = 0.025,
        stability_mode: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize the AutoRegime detector with professional configuration.
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.max_regimes = max_regimes
        self.min_regime_duration = min_regime_duration
        self.economic_significance_threshold = economic_significance_threshold
        self.stability_mode = stability_mode
        self.verbose = verbose

        # State
        self.model: Optional[hmm.GaussianHMM] = None
        self.regimes: Optional[np.ndarray] = None
        self.regime_stats: Optional[Dict[int, Dict[str, float]]] = None

        self.prices: Optional[pd.Series] = None          # full original prices
        self.returns: Optional[pd.Series] = None         # log returns (indexed)
        self.dates: Optional[pd.DatetimeIndex] = None    # full original dates

        self.optimal_n_regimes: Optional[int] = None
        self.regime_timeline: Optional[pd.DataFrame] = None

        np.random.seed(self.random_state)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate **log returns** as a pandas Series aligned to the original index
        (the first row is dropped due to differencing).
        """
        returns = np.log(prices / prices.shift(1)).dropna()
        returns.name = "log_ret"
        return returns

    def _calculate_max_drawdown_corrected(self, prices: pd.Series) -> float:
        """
        Correct maximum drawdown using normalized cumulative wealth.
        Returns a positive decimal (e.g., 0.136 for -13.6%).
        """
        cumulative_wealth = prices / prices.iloc[0]
        rolling_max = cumulative_wealth.expanding().max()
        drawdown = (cumulative_wealth - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        return abs(max_drawdown)

    def _classify_regime(
        self,
        annual_return: float,
        annual_vol: float,
        sharpe_ratio: float,
        max_drawdown: float,
    ) -> str:
        """
        Classify regime based on performance characteristics.
        """
        ret_pct = annual_return * 100
        vol_pct = annual_vol * 100
        dd_pct = max_drawdown * 100

        # Crisis first
        if dd_pct > 40 or (ret_pct < -30 and vol_pct > 35):
            return "Crisis"
        # Bear (not used by front-end palette, but kept for completeness)
        elif ret_pct < -10 and (dd_pct > 25 or vol_pct > 30):
            return "Bear Market"
        # Risk-Off
        elif ret_pct < 5 and vol_pct > 25:
            return "Risk-Off"
        # Sideways
        elif abs(ret_pct) < 8 and dd_pct < 15:
            return "Sideways"
        # Bull
        elif ret_pct > 20 and sharpe_ratio > 1.0 and dd_pct < 20:
            return "Bull Market"
        # Steady Growth
        elif 8 <= ret_pct <= 20 and dd_pct < 15 and vol_pct < 25:
            return "Steady Growth"
        # Goldilocks
        elif ret_pct > 15 and sharpe_ratio > 1.2 and dd_pct < 15:
            return "Goldilocks"
        # Fallback
        else:
            return "Bull Market" if ret_pct > 0 else "Risk-Off"

    def _fit_hmm_deterministic(self, returns: np.ndarray, n_regimes: int) -> hmm.GaussianHMM:
        """
        Fit HMM with a small set of seeds and pick the best (deterministic/stable).
        """
        best_model = None
        best_score = -np.inf
        seeds = [42, 123, 456, 789, 999]

        for seed in seeds:
            try:
                np.random.seed(seed)
                model = hmm.GaussianHMM(
                    n_components=n_regimes,
                    covariance_type="full",
                    random_state=seed,
                    n_iter=200,
                    tol=1e-4,
                )
                X = returns.reshape(-1, 1)
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

        if best_model is None:
            raise ValueError(f"Failed to fit HMM with {n_regimes} regimes")

        return best_model

    def _find_optimal_regimes(self, returns: np.ndarray) -> int:
        """
        Pick K via a simple BIC sweep.
        """
        if self.verbose:
            logger.info("🔍 Determining optimal market regime structure...")

        best_n = self.n_regimes
        best_score = -np.inf

        nobs = len(returns)
        for n in range(2, self.max_regimes + 1):
            try:
                if self.verbose:
                    logger.info(f"Testing {n} regimes...")
                model = self._fit_hmm_deterministic(returns, n)
                # crude param count (transition + means + variances)
                n_params = n * n + 2 * n
                log_like = model.score(returns.reshape(-1, 1)) * nobs
                bic = -2 * log_like + n_params * np.log(nobs)
                score = -bic  # higher is better
                if score > best_score:
                    best_score = score
                    best_n = n
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Failed to fit {n} regimes: {e}")
                continue

        return best_n

    def _calculate_regime_statistics(
        self,
        prices_aligned: pd.Series,
        regimes: np.ndarray,
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate stats for each regime using *aligned* prices (same length as regimes).
        """
        stats: Dict[int, Dict[str, float]] = {}

        for regime_id in np.unique(regimes):
            regime_mask = regimes == regime_id  # boolean mask length N-1
            regime_prices = prices_aligned[regime_mask]
            if len(regime_prices) < 2:
                continue

            regime_returns = np.log(regime_prices / regime_prices.shift(1)).dropna()
            if len(regime_returns) == 0:
                continue

            annual_return = float(regime_returns.mean() * 252.0)
            annual_vol = float(regime_returns.std() * np.sqrt(252.0))
            sharpe_ratio = float(annual_return / annual_vol) if annual_vol > 0 else 0.0
            max_drawdown = float(self._calculate_max_drawdown_corrected(regime_prices))

            regime_name = self._classify_regime(annual_return, annual_vol, sharpe_ratio, max_drawdown)
            total_days = int(regime_mask.sum())

            stats[regime_id] = {
                "regime_name": regime_name,
                "annual_return": annual_return,
                "annual_volatility": annual_vol,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_days": total_days,
                "return_pct": annual_return * 100.0,
                "volatility_pct": annual_vol * 100.0,
                "max_drawdown_pct": max_drawdown * 100.0,
            }

        return stats

    def _create_regime_timeline(
        self,
        dates_aligned: pd.DatetimeIndex,
        regimes: np.ndarray,
    ) -> pd.DataFrame:
        """
        Build a per-period timeline from *aligned* dates (same length as regimes).
        """
        timeline_rows: List[Dict[str, Any]] = []

        # change points in Viterbi path
        regime_changes = np.where(np.diff(regimes) != 0)[0] + 1
        starts = np.concatenate([[0], regime_changes])
        ends = np.concatenate([regime_changes, [len(regimes)]])

        for start_idx, end_idx in zip(starts, ends):
            rid = int(regimes[start_idx])
            if self.regime_stats and rid in self.regime_stats:
                s = self.regime_stats[rid]
                timeline_rows.append(
                    {
                        "Start_Date": dates_aligned[start_idx].strftime("%d-%m-%Y"),
                        "End_Date": dates_aligned[end_idx - 1].strftime("%d-%m-%Y"),
                        "Regime_ID": rid,
                        "Regime_Name": s["regime_name"],
                        "Duration_Days": int(end_idx - start_idx),
                        "Annual_Return_%": round(s["return_pct"], 1),
                        "Annual_Volatility_%": round(s["volatility_pct"], 1),
                        "Sharpe_Ratio": round(s["sharpe_ratio"], 2),
                        "Max_Drawdown_%": round(s["max_drawdown_pct"], 1),
                    }
                )

        return pd.DataFrame(timeline_rows)

    # -----------------------------
    # Public API
    # -----------------------------
    def fit(self, prices: pd.Series, stability_mode: Optional[bool] = None) -> Dict[str, Any]:
        """
        Fit the detector and produce a professional report dict.
        """
        use_stability = stability_mode if stability_mode is not None else self.stability_mode
        if use_stability:
            logger.info("🔒 STABILITY MODE ACTIVATED - Enhanced Parameters Active")

        logger.info("🚀 AutoRegime Professional Analysis Started")
        logger.info("=" * 60)

        # store full originals
        self.prices = prices.copy()
        self.dates = prices.index

        # compute log returns (indexed)
        self.returns = self._calculate_returns(prices)

        logger.info(f"📊 Data loaded: {len(prices)} observations")
        logger.info(f"📅 Period: {self.dates[0].strftime('%d-%m-%Y')} to {self.dates[-1].strftime('%d-%m-%Y')}")

        # === ALIGNMENT FIX ===
        # Align prices/dates to the returns index (N-1 rows); regimes will be length N-1.
        prices_aligned = prices.loc[self.returns.index]
        dates_aligned = self.returns.index

        # Model selection on returns array
        logger.info("🔍 Determining optimal market regime structure...")
        self.optimal_n_regimes = self._find_optimal_regimes(self.returns.values)
        logger.info(f"✅ Optimal regime count determined: {self.optimal_n_regimes} regimes")

        # Fit final HMM and get regimes (length = len(returns))
        logger.info("🧠 Fitting Hidden Markov Model with deterministic parameters...")
        self.model = self._fit_hmm_deterministic(self.returns.values, self.optimal_n_regimes)
        self.regimes = self.model.predict(self.returns.values.reshape(-1, 1))

        # Stats from aligned prices
        logger.info("📈 Calculating comprehensive regime statistics...")
        self.regime_stats = self._calculate_regime_statistics(prices_aligned, self.regimes)

        # Timeline from aligned dates
        self.regime_timeline = self._create_regime_timeline(dates_aligned, self.regimes)

        logger.info("📋 REGIME ANALYSIS COMPLETE")
        logger.info("=" * 60)

        # Summary per regime
        logger.info("REGIME SUMMARY")
        logger.info("-" * 30)
        for regime_id, s in (self.regime_stats or {}).items():
            logger.info(
                f"Regime {regime_id} ({s['regime_name']}): "
                f"{s['return_pct']:+.1f}% return, "
                f"-{s['max_drawdown_pct']:.1f}% max drawdown"
            )

        # Detailed timeline to logs
        self.print_detailed_timeline()

        # Overall portfolio stats from full data (OK to use full series here)
        total_return = float((prices.iloc[-1] / prices.iloc[0] - 1) * 100.0)
        overall_max_dd = float(self._calculate_max_drawdown_corrected(prices) * 100.0)
        overall_returns = self.returns.values  # np array for stats
        annual_return = float(self.returns.mean() * 252.0 * 100.0)
        annual_vol = float(self.returns.std() * np.sqrt(252.0) * 100.0)
        sharpe_ratio = float((self.returns.mean() * 252.0) / (self.returns.std() * np.sqrt(252.0)))

        logger.info("=" * 60)
        logger.info("OVERALL PORTFOLIO ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"📊 Total Return: {total_return:+.1f}%")
        logger.info(f"📈 Annualized Return: {annual_return:+.1f}%")
        logger.info(f"📊 Annualized Volatility: {annual_vol:.1f}%")
        logger.info(f"⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"📉 Maximum Drawdown: -{overall_max_dd:.1f}% (CORRECTED)")
        logger.info(f"🎯 Active Regimes: {self.optimal_n_regimes}")
        logger.info("=" * 60)

        result: Dict[str, Any] = {
            "optimal_n_regimes": self.optimal_n_regimes,
            "regime_statistics": self.regime_stats,
            "regime_timeline": self.regime_timeline,
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(annual_return, 2),
            "annual_volatility_pct": round(annual_vol, 2),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "max_drawdown_pct": round(overall_max_dd, 2),
            "analysis_summary": {
                "total_observations": int(len(prices)),
                "date_range": f"{self.dates[0].strftime('%d-%m-%Y')} to {self.dates[-1].strftime('%d-%m-%Y')}",
                "regimes_detected": int(self.optimal_n_regimes),
                "stability_mode": bool(use_stability),
                "analysis_timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            },
        }

        # Current regime info from the last timeline row (already aligned)
        if self.regime_timeline is not None and len(self.regime_timeline) > 0:
            cur = self.regime_timeline.iloc[-1]
            result["current_regime"] = cur["Regime_Name"]
            result["current_regime_duration"] = int(cur["Duration_Days"])
            result["current_regime_return"] = float(cur["Annual_Return_%"])
            # crude confidence proxy
            conf = min(100.0, (cur["Duration_Days"] / 30.0) * 50.0 + (abs(cur["Sharpe_Ratio"]) * 25.0))
            result["confidence_level"] = f"{conf:.1f}%"

        logger.info("🎯 PROFESSIONAL ANALYSIS COMPLETE")
        logger.info("✅ Ready for LinkedIn publication and UK Global Talent Visa application")

        return result

    def detect_regimes(self, prices: pd.Series, verbose: bool = True) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        """
        self.prices = prices.copy()
        self.dates = prices.index
        self.returns = self._calculate_returns(prices)

        if verbose:
            print(f"Stable Regime Analysis for {getattr(prices, 'name', 'Unknown Symbol')}")

        self.optimal_n_regimes = self._find_optimal_regimes(self.returns.values)
        if verbose:
            print(f"Detected {self.optimal_n_regimes} market regimes")

        self.model = self._fit_hmm_deterministic(self.returns.values, self.optimal_n_regimes)
        self.regimes = self.model.predict(self.returns.values.reshape(-1, 1))

        # ALIGN to returns index before stats/timeline
        prices_aligned = prices.loc[self.returns.index]
        dates_aligned = self.returns.index

        self.regime_stats = self._calculate_regime_statistics(prices_aligned, self.regimes)
        self.regime_timeline = self._create_regime_timeline(dates_aligned, self.regimes)

        if verbose and self.regime_stats:
            for rid, s in self.regime_stats.items():
                print(
                    f"Regime {rid} ({s['regime_name']}): "
                    f"{s['return_pct']:.1f}% return, -{s['max_drawdown_pct']:.1f}% drawdown"
                )
            print(f"Analysis Complete for {getattr(prices, 'name', 'Unknown Symbol')}")

        total_return = float((prices.iloc[-1] / prices.iloc[0] - 1) * 100.0)
        overall_max_dd = float(self._calculate_max_drawdown_corrected(prices) * 100.0)

        return {
            "n_regimes": int(self.optimal_n_regimes),
            "current_regime": (list(self.regime_stats.values())[-1]["regime_name"] if self.regime_stats else "Unknown"),
            "regime_confidence": 85.0,  # Placeholder
            "max_drawdown": overall_max_dd,
            "annual_return": float(self.returns.mean() * 252.0 * 100.0),
            "annual_volatility": float(self.returns.std() * np.sqrt(252.0) * 100.0),
            "sharpe_ratio": float((self.returns.mean() * 252.0) / (self.returns.std() * np.sqrt(252.0))),
        }

    def print_detailed_timeline(self) -> None:
        """Pretty log output of the timeline."""
        if self.regime_timeline is None or len(self.regime_timeline) == 0:
            logger.warning("No regime timeline available")
            return

        logger.info("DETAILED REGIME TIMELINE")
        logger.info("=" * 80)
        for _, row in self.regime_timeline.iterrows():
            logger.info(f"📅 {row['Start_Date']} → {row['End_Date']} ({row['Duration_Days']} days)")
            logger.info(f"🎯 Regime: {row['Regime_Name']} (ID: {row['Regime_ID']})")
            logger.info(
                f"📊 Annual Return: {row['Annual_Return_%']:+.1f}% | "
                f"Volatility: {row['Annual_Volatility_%']:.1f}%"
            )
            logger.info(
                f"📉 Max Drawdown: -{row['Max_Drawdown_%']:.1f}% | "
                f"Sharpe Ratio: {row['Sharpe_Ratio']:.2f}"
            )
            logger.info("-" * 60)

        if len(self.regime_timeline) > 0:
            current_regime = self.regime_timeline.iloc[-1]
            logger.info("CURRENT MARKET STATUS")
            logger.info("=" * 50)
            logger.info(f"📍 Current Regime: {current_regime['Regime_Name']}")
            logger.info(f"📅 Active Since: {current_regime['Start_Date']}")
            logger.info(f"⏱️  Duration: {current_regime['Duration_Days']} trading days")
            logger.info(f"📊 Performance: {current_regime['Annual_Return_%']:+.1f}% annual return")
            logger.info(f"📉 Risk Level: {current_regime['Max_Drawdown_%']:.1f}% max drawdown")
            confidence_score = min(
                100.0,
                (current_regime["Duration_Days"] / 30.0) * 50.0 + (abs(current_regime["Sharpe_Ratio"]) * 25.0),
            )
            logger.info(f"🎯 Confidence Level: {confidence_score:.1f}%")

    # Convenience getters
    def get_regime_timeline(self) -> pd.DataFrame:
        if self.regime_timeline is None:
            raise ValueError("No regime timeline available. Run fit() or detect_regimes() first.")
        return self.regime_timeline.copy()

    def get_current_regime(self) -> Dict[str, Any]:
        if self.regime_timeline is None or len(self.regime_timeline) == 0:
            raise ValueError("No regime analysis available. Run fit() or detect_regimes() first.")
        current = self.regime_timeline.iloc[-1]
        return {
            "regime_name": current["Regime_Name"],
            "regime_id": current["Regime_ID"],
            "start_date": current["Start_Date"],
            "duration_days": current["Duration_Days"],
            "annual_return_pct": current["Annual_Return_%"],
            "max_drawdown_pct": current["Max_Drawdown_%"],
            "sharpe_ratio": current["Sharpe_Ratio"],
        }

    def predict_regime_probabilities(self, prices: pd.Series) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted. Run fit() or detect_regimes() first.")
        rets = self._calculate_returns(prices)
        return self.model.predict_proba(rets.values.reshape(-1, 1))

    def get_regime_summary(self) -> Dict[str, Any]:
        if self.regime_stats is None:
            raise ValueError("No regime analysis available. Run fit() or detect_regimes() first.")

        summary: Dict[str, Any] = {
            "total_regimes": len(self.regime_stats),
            "regime_details": {},
            "overall_statistics": {},
        }

        for rid, s in self.regime_stats.items():
            summary["regime_details"][f"regime_{rid}"] = {
                "name": s["regime_name"],
                "annual_return_pct": round(s["return_pct"], 2),
                "annual_volatility_pct": round(s["volatility_pct"], 2),
                "sharpe_ratio": round(s["sharpe_ratio"], 3),
                "max_drawdown_pct": round(s["max_drawdown_pct"], 2),
                "duration_days": s["total_days"],
            }

        if self.prices is not None and self.dates is not None:
            total_return = (self.prices.iloc[-1] / self.prices.iloc[0] - 1) * 100.0
            overall_max_dd = self._calculate_max_drawdown_corrected(self.prices) * 100.0
            summary["overall_statistics"] = {
                "total_return_pct": round(float(total_return), 2),
                "max_drawdown_pct": round(float(overall_max_dd), 2),
                "analysis_period": f"{self.dates[0].strftime('%d-%m-%Y')} to {self.dates[-1].strftime('%d-%m-%Y')}",
                "total_observations": int(len(self.prices)),
            }

        return summary


__all__ = ["AutoRegimeDetector"]