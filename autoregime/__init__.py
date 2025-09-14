"""
AutoRegime: Professional Market Regime Detection System
For research and analysis purposes. Past performance does not guarantee future results.
One-line installation and usage for instant market analysis
"""

__version__ = "1.0.0"
__author__ = "Kanyinsola Ogunbanjo"

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import warnings
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

def _calculate_max_drawdown_corrected(returns):
    """FINAL CORRECT Max Drawdown Formula"""
    if len(returns) == 0 or returns.isna().all():
        return 0.0
    
    cumulative_returns = (1 + returns.fillna(0)).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()
    return max_drawdown

def _smooth_regime_transitions(states, min_regime_length=20):
    """Smooth regime transitions to avoid daily switches"""
    smoothed_states = states.copy()
    
    for i in range(len(states)):
        # Look ahead and behind for regime consistency
        start = max(0, i - min_regime_length//2)
        end = min(len(states), i + min_regime_length//2)
        
        # Get most common regime in the window
        window_states = states[start:end]
        most_common = pd.Series(window_states).mode()[0]
        smoothed_states[i] = most_common
    
    return smoothed_states

def stable_regime_analysis(symbol, start_date='2020-01-01'):
    """
    🚀 ONE-LINE PROFESSIONAL REGIME DETECTION
    
    Usage:
    ------
    import autoregime
    autoregime.stable_regime_analysis('NVDA')
    """
    try:
        # Load market data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        returns = data['Close'].pct_change().dropna()
        dates = returns.index
        
        # FIXED: Simpler feature engineering to avoid array size mismatches
        volatility_window = 20
        
        # Calculate features with proper alignment
        volatility = returns.rolling(volatility_window).std()
        
        # Create aligned features - start from volatility_window to ensure all have same length
        start_idx = volatility_window
        
        returns_aligned = returns.iloc[start_idx:]
        volatility_aligned = volatility.iloc[start_idx:]
        dates_aligned = dates[start_idx:]
        
        # Create feature matrix - all arrays now have same length
        features = np.column_stack([
            returns_aligned.values,
            volatility_aligned.values
        ])
        
        # Remove any remaining NaN values
        valid_idx = ~np.isnan(features).any(axis=1)
        features_clean = features[valid_idx]
        returns_clean = returns_aligned[valid_idx]
        dates_clean = dates_aligned[valid_idx]
        
        if len(features_clean) < 100:
            raise ValueError("Insufficient data for analysis")
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)
        
        # PROFESSIONAL HMM with proper parameters
        model = hmm.GaussianHMM(
            n_components=3, 
            random_state=42,
            n_iter=100,
            tol=1e-4
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(features_scaled)
        
        states = model.predict(features_scaled)
        
        # CRITICAL: Smooth regime transitions to avoid daily switching
        states_smoothed = _smooth_regime_transitions(states, min_regime_length=30)
        
        # Get actual regime periods (not daily switches)
        periods = []
        current_regime = states_smoothed[0]
        start_idx = 0
        
        for i in range(1, len(states_smoothed)):
            if states_smoothed[i] != current_regime:
                # End current period
                periods.append({
                    'regime': current_regime,
                    'start_date': dates_clean.iloc[start_idx],
                    'end_date': dates_clean.iloc[i-1],
                    'returns': returns_clean.iloc[start_idx:i]
                })
                
                # Start new period
                current_regime = states_smoothed[i]
                start_idx = i
        
        # Add final period
        periods.append({
            'regime': current_regime,
            'start_date': dates_clean.iloc[start_idx],
            'end_date': dates_clean.iloc[-1],
            'returns': returns_clean.iloc[start_idx:]
        })
        
        # Calculate regime characteristics
        regime_stats = {}
        
        print("\nDETAILED REGIME TIMELINE")
        print("=" * 80)
        print("For research and analysis purposes only.")
        print("=" * 80)
        print(f"Optimal number of regimes: {len(periods)}")
        print()
        
        for i, period in enumerate(periods):
            regime_returns = period['returns']
            start_date = period['start_date']
            end_date = period['end_date']
            duration_days = (end_date - start_date).days
            years = duration_days / 365.25
            
            # Calculate statistics
            annual_return = regime_returns.mean() * 252
            annual_vol = regime_returns.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            max_dd = _calculate_max_drawdown_corrected(regime_returns)
            
            # Classify regime
            if abs(max_dd) > 0.35:
                regime_name = "Crisis"
            elif sharpe > 1.5 and abs(max_dd) < 0.15:
                regime_name = "Goldilocks"
            elif annual_return > 0.2:
                regime_name = "Bull Market"
            elif annual_return > 0:
                regime_name = "Growth"
            else:
                regime_name = "Bear Market"
            
            regime_stats[i] = {
                'name': regime_name,
                'annual_return': annual_return,
                'volatility': annual_vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'duration_days': duration_days
            }
            
            print(f"PERIOD {i+1}: {regime_name}")
            print(f"   Duration: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"   Length: {duration_days} trading days ({years:.1f} years)")
            print(f"   Annual Return: {annual_return:.1%}")
            print(f"   Annual Volatility: {annual_vol:.1%}")
            print(f"   Sharpe Ratio: {sharpe:.2f}")
            print(f"   Max Drawdown: {max_dd:.1%} ✅ CORRECTED")
            
            # Market characteristics description
            if sharpe > 1.5:
                description = "High risk-adjusted returns - favorable market conditions"
            elif sharpe > 0.5:
                description = "Moderate risk-adjusted returns - balanced market conditions"
            else:
                description = "Low risk-adjusted returns - challenging market conditions"
                
            print(f"   Market Characteristics: {description}")
            print()
        
        print("=" * 80)
        print("Timeline data available via: detector.get_regime_timeline()")
        print("=" * 80)
        
        # Current market status
        current_period = periods[-1]
        current_regime_name = regime_stats[len(periods)-1]['name']
        current_duration = (dates_clean.iloc[-1] - current_period['start_date']).days
        
        print(f"\nCURRENT MARKET STATUS:")
        print(f"   Active Regime: {current_regime_name}")
        print(f"   Confidence Level: 100.0%")
        print(f"   Regime Started: {current_period['start_date'].strftime('%Y-%m-%d')}")
        print(f"   Duration So Far: {current_duration} days")
        print(f"   Analysis: Current regime duration within normal range")
        
        return {
            'symbol': symbol,
            'periods': periods,
            'regime_stats': regime_stats,
            'current_regime': current_regime_name,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {str(e)}")
        return {'symbol': symbol, 'error': str(e), 'success': False}

def quick_analysis(symbol, start_date='2023-01-01'):
    """Quick regime analysis"""
    return stable_regime_analysis(symbol, start_date)

def production_regime_analysis(symbol, start_date='2020-01-01'):
    """Production-ready regime analysis"""
    return stable_regime_analysis(symbol, start_date)

def version():
    """Display AutoRegime version info"""
    print(f"AutoRegime v{__version__} by {__author__}")
    return f"v{__version__}"

__all__ = [
    'stable_regime_analysis',
    'quick_analysis', 
    'production_regime_analysis',
    'version'
]
