"""
AutoRegime: Professional Market Regime Detection System
For research and analysis purposes. Past performance does not guarantee future results.
Author: [Kanyinsola Ogunbanjo]
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from hmmlearn import hmm
from scipy import stats
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoRegimeDetector:
    """
    Professional automated market regime detection system.
    
    This tool provides historical market analysis for research purposes.
    Past performance does not guarantee future results.
    
    Features:
    - Automatic optimal regime count discovery
    - Real-time regime pattern recognition
    - Multi-timeframe analysis
    - Economic significance testing
    - Professional market regime timeline
    
    Example:
    --------
    >>> detector = AutoRegimeDetector()
    >>> detector.fit(returns_data)
    >>> timeline = detector.get_regime_timeline(returns_data)
    """
    
    def __init__(self, 
                 max_regimes: int = 8,
                 min_regime_duration: int = 5,
                 economic_significance_threshold: float = 0.02,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize AutoRegime detector.
        
        Parameters:
        -----------
        max_regimes : int, default=8
            Maximum number of regimes to test
        min_regime_duration : int, default=5  
            Minimum days a regime must persist
        economic_significance_threshold : float, default=0.02
            Minimum return difference between regimes (2%)
        random_state : int, default=42
            Random seed for reproducibility
        verbose : bool, default=True
            Print progress information
        """
        self.max_regimes = max_regimes
        self.min_regime_duration = min_regime_duration
        self.economic_threshold = economic_significance_threshold
        self.random_state = random_state
        self.verbose = verbose
        
        # Model storage
        self.optimal_model = None
        self.optimal_n_regimes = None
        self.regime_characteristics = {}
        self.regime_names = {}
        self.feature_scaler = StandardScaler()
        
        # Model selection results
        self.model_selection_results = []
        
        # Online learning components
        self.online_learning_enabled = False
        self.adaptation_rate = 0.1
        
    def fit(self, returns_data: pd.DataFrame, 
            feature_columns: Optional[List[str]] = None) -> 'AutoRegimeDetector':
        """
        Fit AutoRegime detector to historical data.
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            Asset returns data with datetime index
        feature_columns : list, optional
            Custom feature columns to use
            
        Returns:
        --------
        self : AutoRegimeDetector
            Fitted detector instance
        """
        logger.info("Starting AutoRegime detection...")
        
        # Validate input data
        self._validate_input_data(returns_data)
        
        # Prepare features for regime detection
        features = self._prepare_features(returns_data, feature_columns)
        
        # Auto-discover optimal number of regimes
        self.optimal_model, self.optimal_n_regimes = self._auto_discover_regimes(
            features, returns_data
        )
        
        # Analyze regime characteristics
        self._analyze_regime_characteristics(returns_data, features)
        
        # Generate regime names
        self._generate_regime_names()
        
        if self.verbose:
            self._print_regime_summary()
            # Show detailed timeline with exact dates
            self.print_detailed_timeline(returns_data)
            
        logger.info("AutoRegime detection completed successfully!")
        return self
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data quality."""
        if len(data) < 100:
            raise ValueError("Need at least 100 observations for reliable regime detection")
        
        if data.isnull().sum().sum() > len(data) * 0.1:
            logger.warning("High percentage of missing values detected")
            
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Index is not datetime - consider converting for better results")
    
    def _prepare_features(self, returns_data: pd.DataFrame, 
                         feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Prepare feature matrix for regime detection.
        
        This is where the magic happens - we create features that capture
        different market regimes effectively.
        """
        if feature_columns is not None:
            # Use custom features
            features = returns_data[feature_columns].values
        else:
            # Create comprehensive feature set
            feature_list = []
            
            # 1. Market return (equal-weighted portfolio)
            market_return = returns_data.mean(axis=1)
            feature_list.append(market_return.values.reshape(-1, 1))
            
            # 2. Market volatility (rolling 21-day)
            market_volatility = returns_data.rolling(21, min_periods=5).std().mean(axis=1)
            market_volatility = market_volatility.fillna(method='bfill').fillna(method='ffill')
            feature_list.append(market_volatility.values.reshape(-1, 1))
            
            # 3. Cross-sectional volatility (dispersion across assets)
            cross_sectional_vol = returns_data.std(axis=1)
            feature_list.append(cross_sectional_vol.values.reshape(-1, 1))
            
            # 4. Momentum (12-month rolling mean)
            momentum = returns_data.rolling(252, min_periods=21).mean().mean(axis=1)
            momentum = momentum.fillna(method='bfill').fillna(method='ffill')
            feature_list.append(momentum.values.reshape(-1, 1))
            
            # 5. Skewness (21-day rolling)
            skewness = returns_data.rolling(21, min_periods=5).skew().mean(axis=1)
            skewness = skewness.fillna(0)
            feature_list.append(skewness.values.reshape(-1, 1))
            
            # 6. Correlation regime indicator
            correlation_indicator = self._calculate_correlation_regime(returns_data)
            feature_list.append(correlation_indicator.reshape(-1, 1))
            
            # Combine all features
            features = np.hstack(feature_list)
        
        # Standardize features
        features = self.feature_scaler.fit_transform(features)
        
        # Handle any remaining NaN values
        features = np.nan_to_num(features, nan=0.0)
        
        return features
    
    def _calculate_correlation_regime(self, returns_data: pd.DataFrame, 
                                    window: int = 63) -> np.ndarray:
        """
        Calculate correlation regime indicator.
        High correlations often indicate crisis/risk-off regimes.
        """
        correlation_values = []
        
        for i in range(len(returns_data)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx >= 10:  # Minimum window
                window_data = returns_data.iloc[start_idx:end_idx]
                corr_matrix = window_data.corr()
                
                # Average off-diagonal correlation
                mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
                avg_correlation = corr_matrix.where(mask).stack().mean()
                correlation_values.append(avg_correlation)
            else:
                correlation_values.append(0.0)
        
        return np.array(correlation_values)
    
    def _auto_discover_regimes(self, features: np.ndarray, 
                              returns_data: pd.DataFrame) -> Tuple[hmm.GaussianHMM, int]:
        """
        Automatically discover optimal number of regimes.
        
        This is the core innovation that makes AutoRegime special.
        """
        logger.info("Auto-discovering optimal regime count...")
        
        best_model = None
        best_score = np.inf
        best_n_regimes = 2
        
        for n_regimes in range(2, self.max_regimes + 1):
            if self.verbose:
                print(f"Testing {n_regimes} regimes...")
            
            try:
                # Fit HMM with current regime count
                model = self._fit_hmm_model(features, n_regimes)
                
                # Calculate comprehensive score
                score_dict = self._calculate_model_score(model, features, returns_data)
                combined_score = score_dict['combined_score']
                
                # Store results
                score_dict['n_regimes'] = n_regimes
                score_dict['model'] = model
                self.model_selection_results.append(score_dict)
                
                # Check if this is the best model
                if combined_score < best_score:
                    best_score = combined_score
                    best_model = model
                    best_n_regimes = n_regimes
                    
            except Exception as e:
                logger.warning(f"Failed to fit model with {n_regimes} regimes: {e}")
                continue
        
        if best_model is None:
            raise ValueError("Failed to fit any regime model")
        
        logger.info(f"Optimal regime count discovered: {best_n_regimes}")
        return best_model, best_n_regimes
    
    def _fit_hmm_model(self, features: np.ndarray, n_regimes: int) -> hmm.GaussianHMM:
        """Fit Hidden Markov Model with specified number of regimes."""
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=1000,
            tol=1e-6,
            random_state=self.random_state
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(features)
            
        return model
    
    def _calculate_model_score(self, model: hmm.GaussianHMM, 
                              features: np.ndarray, 
                              returns_data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive model selection score.
        
        Combines statistical fit with economic significance.
        """
        n_params = model._get_n_fit_scalars_per_param()
        total_params = sum(n_params.values())
        n_obs = len(features)
        
        # Statistical criteria
        log_likelihood = model.score(features)
        aic = -2 * log_likelihood + 2 * total_params
        bic = -2 * log_likelihood + total_params * np.log(n_obs)
        
        # Economic significance
        regime_states = model.predict(features)
        economic_significance = self._test_economic_significance(
            regime_states, returns_data
        )
        
        # Regime persistence (penalize too-frequent switching)
        regime_persistence = self._calculate_regime_persistence(regime_states)
        
        # Combined score (lower is better)
        combined_score = (
            0.4 * bic +  # Statistical fit (40%)
            0.3 * (1 - economic_significance) +  # Economic significance (30%)
            0.3 * (1 - regime_persistence)  # Regime persistence (30%)
        )
        
        return {
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'economic_significance': economic_significance,
            'regime_persistence': regime_persistence,
            'combined_score': combined_score
        }
    
    def _test_economic_significance(self, regime_states: np.ndarray, 
                                   returns_data: pd.DataFrame) -> float:
        """
        Test if regimes have economically significant differences.
        
        Returns score between 0 (no significance) and 1 (highly significant).
        """
        market_returns = returns_data.mean(axis=1).values
        unique_regimes = np.unique(regime_states)
        
        if len(unique_regimes) < 2:
            return 0.0
        
        regime_returns = []
        for regime in unique_regimes:
            regime_mask = regime_states == regime
            if np.sum(regime_mask) > 5:  # Minimum observations
                regime_return_mean = np.mean(market_returns[regime_mask])
                regime_returns.append(regime_return_mean)
        
        if len(regime_returns) < 2:
            return 0.0
        
        # Calculate maximum difference between regimes (annualized)
        max_diff = (np.max(regime_returns) - np.min(regime_returns)) * 252
        
        # Normalize by threshold
        significance_score = min(max_diff / self.economic_threshold, 1.0)
        
        return significance_score
    
    def _calculate_regime_persistence(self, regime_states: np.ndarray) -> float:
        """
        Calculate regime persistence score.
        
        Returns score between 0 (too frequent switching) and 1 (appropriate persistence).
        """
        # Count regime switches
        switches = np.sum(regime_states[1:] != regime_states[:-1])
        
        # Calculate average regime duration
        avg_duration = len(regime_states) / (switches + 1)
        
        # Score based on minimum duration requirement
        persistence_score = min(avg_duration / self.min_regime_duration, 1.0)
        
        return persistence_score
    
    def _analyze_regime_characteristics(self, returns_data: pd.DataFrame, 
                                      features: np.ndarray) -> None:
        """Analyze and store characteristics of each regime."""
        regime_states = self.optimal_model.predict(features)
        market_returns = returns_data.mean(axis=1)
        
        self.regime_characteristics = {}
        
        for regime in range(self.optimal_n_regimes):
            regime_mask = regime_states == regime
            
            if np.sum(regime_mask) > 0:
                regime_returns = market_returns[regime_mask]
                regime_features = features[regime_mask]
                
                self.regime_characteristics[regime] = {
                    'frequency': np.sum(regime_mask) / len(regime_states),
                    'avg_duration': self._calculate_avg_duration(regime_states, regime),
                    'mean_return': np.mean(regime_returns) * 252,  # Annualized
                    'volatility': np.std(regime_returns) * np.sqrt(252),  # Annualized
                    'sharpe_ratio': (np.mean(regime_returns) * 252) / (np.std(regime_returns) * np.sqrt(252)),
                    'max_drawdown': self._calculate_max_drawdown(regime_returns),
                    'feature_means': np.mean(regime_features, axis=0)
                }
    
    def _calculate_avg_duration(self, regime_states: np.ndarray, regime: int) -> float:
        """Calculate average duration of a specific regime."""
        durations = []
        current_duration = 0
        
        for state in regime_states:
            if state == regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        # Don't forget the last duration
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown for a return series."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative / running_max) - 1
        return np.min(drawdown)
    
    def _generate_regime_names(self) -> None:
        """Generate intuitive names for regimes based on characteristics."""
        self.regime_names = {}
        
        # Sort regimes by return/risk characteristics
        regime_data = []
        for regime in range(self.optimal_n_regimes):
            if regime in self.regime_characteristics:
                char = self.regime_characteristics[regime]
                regime_data.append({
                    'regime': regime,
                    'return': char['mean_return'],
                    'volatility': char['volatility'],
                    'sharpe': char['sharpe_ratio']
                })
        
        # Sort by Sharpe ratio (best to worst)
        regime_data.sort(key=lambda x: x['sharpe'], reverse=True)
        
        # Assign names based on risk/return profile
        name_templates = [
            "Goldilocks",      # High Sharpe
            "Bull Market",     # Good performance
            "Steady Growth",   # Moderate performance
            "Sideways",        # Neutral
            "Risk-Off",        # Poor performance
            "Bear Market",     # Worst performance
        ]
        
        for i, regime_info in enumerate(regime_data):
            regime_num = regime_info['regime']
            name_idx = min(i, len(name_templates) - 1)
            
            self.regime_names[regime_num] = name_templates[name_idx]
    
    def _print_regime_summary(self) -> None:
        """Print comprehensive regime analysis summary."""
        print("\n" + "="*60)
        print("AUTOREGIME ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Optimal number of regimes: {self.optimal_n_regimes}")
        print(f"Model selection score: {self.model_selection_results[-1]['combined_score']:.3f}")
        
        print("\n REGIME CHARACTERISTICS:")
        print("-" * 60)
        
        for regime in range(self.optimal_n_regimes):
            if regime in self.regime_characteristics:
                char = self.regime_characteristics[regime]
                name = self.regime_names.get(regime, f"Regime {regime}")
                
                print(f"\n{name} (Regime {regime}):")
                print(f"  Frequency: {char['frequency']:.1%}")
                print(f"  Avg Duration: {char['avg_duration']:.1f} days")
                print(f"  Annual Return: {char['mean_return']:.1%}")
                print(f"  Annual Volatility: {char['volatility']:.1%}")
                print(f"  Sharpe Ratio: {char['sharpe_ratio']:.2f}")
                print(f"  Max Drawdown: {char['max_drawdown']:.1%}")
        
        print("\n" + "="*60)
    
    def get_regime_timeline(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed regime timeline with exact dates for professional analysis.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data used for fitting
            
        Returns:
        --------
        pd.DataFrame: Timeline with regime periods, dates, and characteristics
        """
        if self.optimal_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get regime predictions
        features = self._prepare_features(data)
        regimes = self.optimal_model.predict(features)
        
        regime_periods = []
        
        if len(regimes) == 0:
            return pd.DataFrame()
        
        # Track regime changes
        current_regime = regimes[0]
        start_date = data.index[0]
        start_idx = 0
        
        for i in range(1, len(regimes)):
            if regimes[i] != current_regime:
                # End of current regime period
                end_date = data.index[i-1]
                duration = i - start_idx
                
                # Get regime characteristics
                regime_name = self.regime_names.get(current_regime, f'Regime {current_regime}')
                regime_char = self.regime_characteristics.get(current_regime, {})
                
                period_info = {
                    'Regime_ID': current_regime,
                    'Regime_Name': regime_name,
                    'Start_Date': start_date,
                    'End_Date': end_date,
                    'Duration_Days': duration,
                    'Duration_Years': duration / 252,
                    'Annual_Return_Pct': regime_char.get('mean_return', 0) * 100,
                    'Annual_Volatility_Pct': regime_char.get('volatility', 0) * 100,
                    'Sharpe_Ratio': regime_char.get('sharpe_ratio', 0),
                    'Max_Drawdown_Pct': regime_char.get('max_drawdown', 0) * 100
                }
                regime_periods.append(period_info)
                
                # Start new regime period
                current_regime = regimes[i]
                start_date = data.index[i]
                start_idx = i
        
        # Add the last period
        end_date = data.index[-1]
        duration = len(regimes) - start_idx
        regime_name = self.regime_names.get(current_regime, f'Regime {current_regime}')
        regime_char = self.regime_characteristics.get(current_regime, {})
        
        period_info = {
            'Regime_ID': current_regime,
            'Regime_Name': regime_name,
            'Start_Date': start_date,
            'End_Date': end_date,
            'Duration_Days': duration,
            'Duration_Years': duration / 252,
            'Annual_Return_Pct': regime_char.get('mean_return', 0) * 100,
            'Annual_Volatility_Pct': regime_char.get('volatility', 0) * 100,
            'Sharpe_Ratio': regime_char.get('sharpe_ratio', 0),
            'Max_Drawdown_Pct': regime_char.get('max_drawdown', 0) * 100
        }
        regime_periods.append(period_info)
        
        timeline_df = pd.DataFrame(regime_periods)
        return timeline_df

    def print_detailed_timeline(self, data: pd.DataFrame) -> None:
        """Print detailed regime timeline with exact dates and market characteristics."""
        timeline = self.get_regime_timeline(data)
        
        print("\nDETAILED REGIME TIMELINE")
        print("="*80)
        print("For research and analysis purposes only.")
        print("="*80)
        
        for idx, period in timeline.iterrows():
            print(f"\nPERIOD {idx + 1}: {period['Regime_Name']}")
            print(f"   Duration: {period['Start_Date'].strftime('%Y-%m-%d')} to {period['End_Date'].strftime('%Y-%m-%d')}")
            print(f"   Length: {period['Duration_Days']} trading days ({period['Duration_Years']:.1f} years)")
            print(f"   Annual Return: {period['Annual_Return_Pct']:.1f}%")
            print(f"   Annual Volatility: {period['Annual_Volatility_Pct']:.1f}%")
            print(f"   Sharpe Ratio: {period['Sharpe_Ratio']:.2f}")
            print(f"   Max Drawdown: {period['Max_Drawdown_Pct']:.1f}%")
            
            # Market regime characteristics
            if period['Sharpe_Ratio'] > 1.0:
                characteristics = "High risk-adjusted returns - favorable market conditions"
            elif period['Sharpe_Ratio'] > 0.5:
                characteristics = "Moderate risk-adjusted returns - balanced market conditions"
            elif period['Sharpe_Ratio'] > 0:
                characteristics = "Positive returns with elevated risk - mixed conditions"
            else:
                characteristics = "Challenging period - poor risk-adjusted performance"
            
            print(f"   Market Characteristics: {characteristics}")
        
        print("\n" + "="*80)
        
        # Export timeline automatically
        timeline.to_csv('autoregime_timeline.csv', index=False)
        print("Timeline exported to 'autoregime_timeline.csv'")
        
        # Current regime status
        current_regime, confidence = self.predict_current_regime(data.tail(21))
        current_name = self.regime_names.get(current_regime, f'Regime {current_regime}')
        current_period = timeline.iloc[-1]
        
        print(f"\nCURRENT MARKET STATUS:")
        print(f"   Active Regime: {current_name}")
        print(f"   Confidence Level: {confidence:.1%}")
        print(f"   Regime Started: {current_period['Start_Date'].strftime('%Y-%m-%d')}")
        print(f"   Duration So Far: {current_period['Duration_Days']} days")
        
        # Regime stability analysis
        avg_duration = timeline['Duration_Days'].mean()
        if current_period['Duration_Days'] > avg_duration * 1.5:
            print(f"   Analysis: Current regime duration exceeds historical average")
        else:
            print(f"   Analysis: Current regime duration within normal range")
    
    def predict_current_regime(self, recent_data: Optional[pd.DataFrame] = None, 
                              window: int = 21) -> Tuple[int, float]:
        """
        Predict current market regime.
        
        Parameters:
        -----------
        recent_data : pd.DataFrame, optional
            Recent market data for prediction
        window : int, default=21
            Lookback window for prediction
            
        Returns:
        --------
        regime : int
            Predicted regime number
        confidence : float
            Prediction confidence (0-1)
        """
        if self.optimal_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if recent_data is None:
            raise ValueError("Must provide recent_data for prediction")
        
        # Prepare features for recent data
        recent_features = self._prepare_features(recent_data.tail(window))
        
        # Get regime probabilities
        regime_probs = self.optimal_model.predict_proba(recent_features)
        
        # Use most recent probability
        latest_probs = regime_probs[-1]
        predicted_regime = np.argmax(latest_probs)
        confidence = np.max(latest_probs)
        
        return predicted_regime, confidence
    
    def get_regime_probabilities(self, data: pd.DataFrame) -> np.ndarray:
        """Get regime probabilities for given data."""
        if self.optimal_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        features = self._prepare_features(data)
        return self.optimal_model.predict_proba(features)
    
    def get_regime_summary(self) -> Dict:
        """Get comprehensive summary of regime analysis."""
        if self.optimal_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return {
            'optimal_n_regimes': self.optimal_n_regimes,
            'regime_names': self.regime_names,
            'regime_characteristics': self.regime_characteristics,
            'model_selection_results': self.model_selection_results
        }
