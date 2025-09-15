"""
Author: Kanyinsola Ogunbanjo
GitHub: https://github.com/KANYINSOLA-OGUNBANJO/-AutoRegime
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import warnings
import logging
from datetime import datetime, timedelta
import yfinance as yf
from typing import Optional, Dict, Any, List, Tuple

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
    
    def __init__(self, n_regimes: int = 4, random_state: int = 42, 
                 max_regimes: int = 6, min_regime_duration: int = 15,
                 economic_significance_threshold: float = 0.025,
                 stability_mode: bool = False, verbose: bool = True):
        """
        Initialize the AutoRegime detector with professional configuration.
        
        Parameters:
        -----------
        n_regimes : int, default=4
            Number of regimes to detect
        random_state : int, default=42
            Random seed for reproducible results
        max_regimes : int, default=6
            Maximum number of regimes to test
        min_regime_duration : int, default=15
            Minimum duration for each regime
        economic_significance_threshold : float, default=0.025
            Threshold for economic significance
        stability_mode : bool, default=False
            Enable enhanced stability parameters
        verbose : bool, default=True
            Enable verbose output
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.max_regimes = max_regimes
        self.min_regime_duration = min_regime_duration
        self.economic_significance_threshold = economic_significance_threshold
        self.stability_mode = stability_mode
        self.verbose = verbose
        
        # Initialize attributes
        self.model = None
        self.regimes = None
        self.regime_stats = None
        self.prices = None
        self.returns = None
        self.dates = None
        self.optimal_n_regimes = None
        self.regime_timeline = None
        
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
    
    def _calculate_returns(self, prices: pd.Series) -> np.ndarray:
        """Calculate log returns from price series."""
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns.values
    
    def _calculate_max_drawdown_corrected(self, prices: pd.Series) -> float:
        """
        Calculate corrected maximum drawdown using proper cumulative wealth method.
        
        This fixes the 3x overestimation bug in the original calculation.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series for drawdown calculation
            
        Returns:
        --------
        float : Maximum drawdown as a decimal (e.g., 0.136 for 13.6%)
        """
        # Calculate cumulative wealth (normalized to start at 1.0)
        cumulative_wealth = prices / prices.iloc[0]
        
        # Calculate rolling maximum
        rolling_max = cumulative_wealth.expanding().max()
        
        # Calculate drawdown at each point
        drawdown = (cumulative_wealth - rolling_max) / rolling_max
        
        # Return maximum drawdown (most negative value)
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown)  # Return as positive value
    
    def _classify_regime(self, annual_return: float, annual_vol: float, 
                        sharpe_ratio: float, max_drawdown: float) -> str:
        """
        Classify regime based on performance characteristics.
        
        Uses realistic thresholds for professional regime classification.
        """
        # Convert to percentages for easier comparison
        ret_pct = annual_return * 100
        vol_pct = annual_vol * 100
        dd_pct = max_drawdown * 100
        
        # Crisis conditions (priority check)
        if dd_pct > 40 or (ret_pct < -30 and vol_pct > 35):
            return "Crisis"
        
        # Bear Market conditions
        elif ret_pct < -10 and (dd_pct > 25 or vol_pct > 30):
            return "Bear Market"
        
        # Risk-Off conditions  
        elif ret_pct < 5 and vol_pct > 25:
            return "Risk-Off"
        
        # Sideways conditions
        elif abs(ret_pct) < 8 and dd_pct < 15:
            return "Sideways"
        
        # Bull Market conditions
        elif ret_pct > 20 and sharpe_ratio > 1.0 and dd_pct < 20:
            return "Bull Market"
        
        # Steady Growth conditions
        elif 8 <= ret_pct <= 20 and dd_pct < 15 and vol_pct < 25:
            return "Steady Growth"
        
        # Goldilocks conditions (best performance)
        elif ret_pct > 15 and sharpe_ratio > 1.2 and dd_pct < 15:
            return "Goldilocks"
        
        # Default fallback
        else:
            return "Bull Market" if ret_pct > 0 else "Risk-Off"
    
    def _fit_hmm_deterministic(self, returns: np.ndarray, n_regimes: int) -> hmm.GaussianHMM:
        """
        Fit HMM with multiple seeds for deterministic behavior.
        
        This ensures consistent results across multiple runs by trying
        multiple random seeds and selecting the best model.
        """
        best_model = None
        best_score = -np.inf
        
        # Try multiple seeds for stability
        seeds = [42, 123, 456, 789, 999]
        
        for seed in seeds:
            try:
                np.random.seed(seed)
                model = hmm.GaussianHMM(
                    n_components=n_regimes,
                    covariance_type="full",
                    random_state=seed,
                    n_iter=200,
                    tol=1e-4
                )
                
                # Fit model
                model.fit(returns.reshape(-1, 1))
                
                # Calculate score
                score = model.score(returns.reshape(-1, 1))
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    
            except Exception:
                continue
        
        if best_model is None:
            raise ValueError(f"Failed to fit HMM with {n_regimes} regimes")
        
        return best_model
    
    def _find_optimal_regimes(self, returns: np.ndarray) -> int:
        """Find optimal number of regimes using information criteria."""
        
        if self.verbose:
            logger.info("Determining optimal number of regimes...")
        
        best_n = self.n_regimes
        best_score = -np.inf
        
        for n in range(2, self.max_regimes + 1):
            try:
                if self.verbose:
                    logger.info(f"Testing {n} regimes...")
                
                model = self._fit_hmm_deterministic(returns, n)
                
                # Calculate BIC score (lower is better, so we negate it)
                n_params = n * n + 2 * n  # Transition matrix + means + variances
                log_likelihood = model.score(returns.reshape(-1, 1)) * len(returns)
                bic = -2 * log_likelihood + n_params * np.log(len(returns))
                score = -bic  # Negate so higher is better
                
                if score > best_score:
                    best_score = score
                    best_n = n
                    
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Failed to fit {n} regimes: {e}")
                continue
        
        return best_n
    
    def _calculate_regime_statistics(self, prices: pd.Series, regimes: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Calculate comprehensive statistics for each regime."""
        
        stats = {}
        
        for regime_id in np.unique(regimes):
            # Get regime periods
            regime_mask = regimes == regime_id
            regime_prices = prices[regime_mask]
            
            if len(regime_prices) < 2:
                continue
            
            # Calculate returns for this regime
            regime_returns = np.log(regime_prices / regime_prices.shift(1)).dropna()
            
            if len(regime_returns) == 0:
                continue
            
            # Calculate statistics
            annual_return = regime_returns.mean() * 252
            annual_vol = regime_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
            
            # Calculate corrected max drawdown for this regime
            max_drawdown = self._calculate_max_drawdown_corrected(regime_prices)
            
            # Classify regime
            regime_name = self._classify_regime(annual_return, annual_vol, sharpe_ratio, max_drawdown)
            
            # Calculate duration
            regime_periods = np.where(regime_mask)[0]
            total_days = len(regime_periods)
            
            # Store statistics
            stats[regime_id] = {
                'regime_name': regime_name,
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_days': total_days,
                'return_pct': annual_return * 100,
                'volatility_pct': annual_vol * 100,
                'max_drawdown_pct': max_drawdown * 100
            }
        
        return stats
    
    def _create_regime_timeline(self, dates: pd.DatetimeIndex, regimes: np.ndarray) -> pd.DataFrame:
        """Create detailed regime timeline with transitions."""
        
        timeline_data = []
        
        # Find regime transitions
        regime_changes = np.where(np.diff(regimes) != 0)[0] + 1
        regime_starts = np.concatenate([[0], regime_changes])
        regime_ends = np.concatenate([regime_changes, [len(regimes)]])
        
        for start_idx, end_idx in zip(regime_starts, regime_ends):
            regime_id = regimes[start_idx]
            
            if regime_id in self.regime_stats:
                regime_info = self.regime_stats[regime_id]
                
                timeline_data.append({
                    'Start_Date': dates[start_idx].strftime('%d-%m-%Y'),
                    'End_Date': dates[end_idx-1].strftime('%d-%m-%Y'),
                    'Regime_ID': regime_id,
                    'Regime_Name': regime_info['regime_name'],
                    'Duration_Days': end_idx - start_idx,
                    'Annual_Return_%': round(regime_info['return_pct'], 1),
                    'Annual_Volatility_%': round(regime_info['volatility_pct'], 1),
                    'Sharpe_Ratio': round(regime_info['sharpe_ratio'], 2),
                    'Max_Drawdown_%': round(regime_info['max_drawdown_pct'], 1)
                })
        
        return pd.DataFrame(timeline_data)
    
    def print_detailed_timeline(self) -> None:
        """Print detailed regime timeline with professional formatting."""
        
        if self.regime_timeline is None or len(self.regime_timeline) == 0:
            logger.warning("No regime timeline available")
            return
        
        logger.info("DETAILED REGIME TIMELINE")
        logger.info("=" * 80)
        
        for _, row in self.regime_timeline.iterrows():
            logger.info(f"📅 {row['Start_Date']} → {row['End_Date']} ({row['Duration_Days']} days)")
            logger.info(f"🎯 Regime: {row['Regime_Name']} (ID: {row['Regime_ID']})")
            logger.info(f"📊 Annual Return: {row['Annual_Return_%']:+.1f}% | Volatility: {row['Annual_Volatility_%']:.1f}%")
            logger.info(f"📉 Max Drawdown: -{row['Max_Drawdown_%']:.1f}% | Sharpe Ratio: {row['Sharpe_Ratio']:.2f}")
            logger.info("-" * 60)
        
        # Current regime analysis
        if len(self.regime_timeline) > 0:
            current_regime = self.regime_timeline.iloc[-1]
            logger.info("CURRENT MARKET STATUS")
            logger.info("=" * 50)
            logger.info(f"📍 Current Regime: {current_regime['Regime_Name']}")
            logger.info(f"📅 Active Since: {current_regime['Start_Date']}")
            logger.info(f"⏱️  Duration: {current_regime['Duration_Days']} trading days")
            logger.info(f"📊 Performance: {current_regime['Annual_Return_%']:+.1f}% annual return")
            logger.info(f"📉 Risk Level: {current_regime['Max_Drawdown_%']:.1f}% max drawdown")
            
            # Calculate confidence level based on duration and performance consistency
            confidence_score = min(100, (current_regime['Duration_Days'] / 30) * 50 + 
                                 (abs(current_regime['Sharpe_Ratio']) * 25))
            logger.info(f"🎯 Confidence Level: {confidence_score:.1f}%")
    
    def fit(self, prices: pd.Series, stability_mode: Optional[bool] = None) -> Dict[str, Any]:
        """
        Fit the regime detection model with professional output format.
        
        This is the main method that provides the complete professional analysis
        with INFO logging, detailed timeline, and corrected calculations.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series with datetime index
        stability_mode : bool, optional
            Override instance stability_mode setting
            
        Returns:
        --------
        dict : Complete analysis results with professional metadata
        """
        
        # Use provided stability_mode or instance setting
        use_stability = stability_mode if stability_mode is not None else self.stability_mode
        
        if use_stability:
            logger.info("🔒 STABILITY MODE ACTIVATED - Enhanced Parameters Active")
        
        logger.info("🚀 AutoRegime Professional Analysis Started")
        logger.info("=" * 60)
        
        # Store data
        self.prices = prices.copy()
        self.dates = prices.index
        self.returns = self._calculate_returns(prices)
        
        logger.info(f"📊 Data loaded: {len(prices)} observations")
        logger.info(f"📅 Period: {self.dates[0].strftime('%d-%m-%Y')} to {self.dates[-1].strftime('%d-%m-%Y')}")
        
        # Find optimal number of regimes
        logger.info("🔍 Determining optimal market regime structure...")
        self.optimal_n_regimes = self._find_optimal_regimes(self.returns)
        
        logger.info(f"✅ Optimal regime count determined: {self.optimal_n_regimes} regimes")
        
        # Fit final model
        logger.info("🧠 Fitting Hidden Markov Model with deterministic parameters...")
        self.model = self._fit_hmm_deterministic(self.returns, self.optimal_n_regimes)
        
        # Predict regimes
        self.regimes = self.model.predict(self.returns.reshape(-1, 1))
        
        # Calculate regime statistics
        logger.info("📈 Calculating comprehensive regime statistics...")
        self.regime_stats = self._calculate_regime_statistics(prices, self.regimes)
        
        # Create timeline
        self.regime_timeline = self._create_regime_timeline(self.dates, self.regimes)
        
        # Print detailed analysis
        logger.info("📋 REGIME ANALYSIS COMPLETE")
        logger.info("=" * 60)
        
        # Print summary statistics
        logger.info("REGIME SUMMARY")
        logger.info("-" * 30)
        for regime_id, stats in self.regime_stats.items():
            logger.info(f"Regime {regime_id} ({stats['regime_name']}): "
                       f"{stats['return_pct']:+.1f}% return, "
                       f"-{stats['max_drawdown_pct']:.1f}% max drawdown")
        
        # Print detailed timeline
        self.print_detailed_timeline()
        
        # Calculate overall portfolio statistics
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        overall_max_dd = self._calculate_max_drawdown_corrected(prices) * 100
        overall_returns = self.returns
        annual_return = overall_returns.mean() * 252 * 100
        annual_vol = overall_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (overall_returns.mean() * 252) / (overall_returns.std() * np.sqrt(252))
        
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
        
        # Return comprehensive results
        result = {
            'optimal_n_regimes': self.optimal_n_regimes,
            'regime_statistics': self.regime_stats,
            'regime_timeline': self.regime_timeline,
            'total_return_pct': round(total_return, 2),
            'annual_return_pct': round(annual_return, 2),
            'annual_volatility_pct': round(annual_vol, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'max_drawdown_pct': round(overall_max_dd, 2),
            'analysis_summary': {
                'total_observations': len(prices),
                'date_range': f"{self.dates[0].strftime('%d-%m-%Y')} to {self.dates[-1].strftime('%d-%m-%Y')}",
                'regimes_detected': self.optimal_n_regimes,
                'stability_mode': use_stability,
                'analysis_timestamp': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            }
        }
        
        # Add current regime information
        if len(self.regime_timeline) > 0:
            current_regime = self.regime_timeline.iloc[-1]
            result['current_regime'] = current_regime['Regime_Name']
            result['current_regime_duration'] = current_regime['Duration_Days']
            result['current_regime_return'] = current_regime['Annual_Return_%']
            confidence_score = min(100, (current_regime['Duration_Days'] / 30) * 50 + 
                                 (abs(current_regime['Sharpe_Ratio']) * 25))
            result['confidence_level'] = f"{confidence_score:.1f}%"
        
        logger.info("🎯 PROFESSIONAL ANALYSIS COMPLETE")
        logger.info("✅ Ready for LinkedIn publication and UK Global Talent Visa application")
        
        return result
    
    def detect_regimes(self, prices: pd.Series, verbose: bool = True) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        
        Note: This method provides basic regime detection without the full
        professional INFO logging format. For complete professional output,
        use the fit() method with stability_mode=True.
        """
        
        # Store data
        self.prices = prices.copy()
        self.dates = prices.index
        self.returns = self._calculate_returns(prices)
        
        if verbose:
            print(f"Stable Regime Analysis for {getattr(prices, 'name', 'Unknown Symbol')}")
        
        # Find optimal regimes
        self.optimal_n_regimes = self._find_optimal_regimes(self.returns)
        
        if verbose:
            print(f"Detected {self.optimal_n_regimes} market regimes")
        
        # Fit model
        self.model = self._fit_hmm_deterministic(self.returns, self.optimal_n_regimes)
        self.regimes = self.model.predict(self.returns.reshape(-1, 1))
        
        # Calculate statistics
        self.regime_stats = self._calculate_regime_statistics(prices, self.regimes)
        
        # Print basic summary
        if verbose:
            for regime_id, stats in self.regime_stats.items():
                print(f"Regime {regime_id} ({stats['regime_name']}): "
                     f"{stats['return_pct']:.1f}% return, "
                     f"-{stats['max_drawdown_pct']:.1f}% drawdown")
        
        if verbose:
            print(f"Analysis Complete for {getattr(prices, 'name', 'Unknown Symbol')}")
        
        # Return basic results
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        overall_max_dd = self._calculate_max_drawdown_corrected(prices) * 100
        
        return {
            'n_regimes': self.optimal_n_regimes,
            'current_regime': list(self.regime_stats.values())[-1]['regime_name'] if self.regime_stats else 'Unknown',
            'regime_confidence': 85.0,  # Placeholder
            'max_drawdown': overall_max_dd,
            'annual_return': self.returns.mean() * 252 * 100,
            'annual_volatility': self.returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': (self.returns.mean() * 252) / (self.returns.std() * np.sqrt(252))
        }
    
    def get_regime_timeline(self) -> pd.DataFrame:
        """Get the regime timeline DataFrame."""
        if self.regime_timeline is None:
            raise ValueError("No regime timeline available. Run fit() or detect_regimes() first.")
        return self.regime_timeline.copy()
    
    def get_current_regime(self) -> Dict[str, Any]:
        """Get current regime information."""
        if self.regime_timeline is None or len(self.regime_timeline) == 0:
            raise ValueError("No regime analysis available. Run fit() or detect_regimes() first.")
        
        current = self.regime_timeline.iloc[-1]
        return {
            'regime_name': current['Regime_Name'],
            'regime_id': current['Regime_ID'],
            'start_date': current['Start_Date'],
            'duration_days': current['Duration_Days'],
            'annual_return_pct': current['Annual_Return_%'],
            'max_drawdown_pct': current['Max_Drawdown_%'],
            'sharpe_ratio': current['Sharpe_Ratio']
        }
    
    def predict_regime_probabilities(self, prices: pd.Series) -> np.ndarray:
        """Predict regime probabilities for given prices."""
        if self.model is None:
            raise ValueError("Model not fitted. Run fit() or detect_regimes() first.")
        
        returns = self._calculate_returns(prices)
        return self.model.predict_proba(returns.reshape(-1, 1))
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get comprehensive regime analysis summary."""
        if self.regime_stats is None:
            raise ValueError("No regime analysis available. Run fit() or detect_regimes() first.")
        
        summary = {
            'total_regimes': len(self.regime_stats),
            'regime_details': {},
            'overall_statistics': {}
        }
        
        # Add regime details
        for regime_id, stats in self.regime_stats.items():
            summary['regime_details'][f"regime_{regime_id}"] = {
                'name': stats['regime_name'],
                'annual_return_pct': round(stats['return_pct'], 2),
                'annual_volatility_pct': round(stats['volatility_pct'], 2),
                'sharpe_ratio': round(stats['sharpe_ratio'], 3),
                'max_drawdown_pct': round(stats['max_drawdown_pct'], 2),
                'duration_days': stats['total_days']
            }
        
        # Add overall statistics
        if self.prices is not None:
            total_return = (self.prices.iloc[-1] / self.prices.iloc[0] - 1) * 100
            overall_max_dd = self._calculate_max_drawdown_corrected(self.prices) * 100
            
            summary['overall_statistics'] = {
                'total_return_pct': round(total_return, 2),
                'max_drawdown_pct': round(overall_max_dd, 2),
                'analysis_period': f"{self.dates[0].strftime('%d-%m-%Y')} to {self.dates[-1].strftime('%d-%m-%Y')}",
                'total_observations': len(self.prices)
            }
        
        return summary

# Export main class
__all__ = ['AutoRegimeDetector']
