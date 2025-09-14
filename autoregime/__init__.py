"""
AutoRegime: Professional Market Regime Detection System
For research and analysis purposes. Past performance does not guarantee future results.
One-line installation and usage for instant market analysis
"""

__version__ = "1.0.0"
__author__ = "Kanyinsola Ogunbanjo"

# Direct imports - NO MODULE DEPENDENCIES THAT BREAK
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import warnings
from typing import Dict, Any, Optional

def stable_regime_analysis(symbol, start_date='2020-01-01'):
    """
    🚀 ONE-LINE MARKET REGIME DETECTION
    
    Revolutionary AI-powered regime analysis that replaces 30+ lines of competitor code.
    
    Usage:
    ------
    import autoregime
    autoregime.stable_regime_analysis('NVDA')
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (e.g., 'AAPL', 'SPY', 'NVDA')
    start_date : str, default='2020-01-01'
        Start date for analysis
        
    Returns:
    --------
    dict : Analysis results with corrected max drawdown calculations
    """
    print(f"🔧 AutoRegime Analysis: {symbol}")
    print("Professional AI-powered regime detection...")
    
    try:
        # Load market data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        returns = data['Close'].pct_change().dropna()
        
        # AI feature engineering
        volatility = returns.rolling(20).std().fillna(returns.std())
        features = StandardScaler().fit_transform(
            np.column_stack([returns.values, volatility.values])
        )
        
        # Hidden Markov Model regime detection
        model = hmm.GaussianHMM(n_components=3, random_state=42, n_iter=100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(features)
        
        states = model.predict(features)
        
        print(f"\n📊 REGIME DETECTION RESULTS:")
        print("="*50)
        
        regime_results = {}
        
        for regime in range(3):
            mask = states == regime
            if np.sum(mask) > 10:  # Minimum data points
                regime_returns = returns[mask]
                
                # ✅ CORRECTED MAX DRAWDOWN CALCULATION
                cumulative = (1 + regime_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdowns = (cumulative - running_max) / running_max
                max_dd = drawdowns.min()
                
                # Regime characteristics
                annual_return = regime_returns.mean() * 252
                annual_vol = regime_returns.std() * np.sqrt(252)
                sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                frequency = np.sum(mask) / len(returns)
                
                # Regime classification
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
                
                regime_results[regime] = {
                    'name': regime_name,
                    'frequency': frequency,
                    'annual_return': annual_return,
                    'volatility': annual_vol,
                    'max_drawdown': max_dd,
                    'sharpe_ratio': sharpe
                }
                
                print(f"\n{regime_name} (Regime {regime}):")
                print(f"  📈 Annual Return: {annual_return:.1%}")
                print(f"  📉 Max Drawdown: {max_dd:.1%} ✅ CORRECTED")
                print(f"  📊 Volatility: {annual_vol:.1%}")
                print(f"  🎯 Sharpe Ratio: {sharpe:.2f}")
                print(f"  ⏱️  Frequency: {frequency:.1%}")
        
        print(f"\n🎉 AutoRegime Analysis Complete!")
        print(f"✅ Professional regime detection in ONE LINE!")
        
        return {
            'symbol': symbol,
            'regimes': regime_results,
            'total_regimes': len(regime_results),
            'states': states,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {str(e)}")
        return {'symbol': symbol, 'error': str(e), 'success': False}

def quick_analysis(symbol, start_date='2023-01-01'):
    """
    Quick regime analysis - simplified version
    
    Usage:
    ------
    import autoregime
    autoregime.quick_analysis('SPY')
    """
    return stable_regime_analysis(symbol, start_date)

def production_regime_analysis(symbol, start_date='2020-01-01'):
    """
    Production-ready regime analysis with maximum stability
    
    Usage:
    ------
    import autoregime
    autoregime.production_regime_analysis('AAPL')
    """
    print(f"🏭 Production Analysis: {symbol}")
    return stable_regime_analysis(symbol, start_date)

def version():
    """Display AutoRegime version info"""
    print(f"AutoRegime v{__version__} by {__author__}")
    print("Revolutionary one-line market regime detection")
    print("Usage: autoregime.stable_regime_analysis('SYMBOL')")

# Export the one-line APIs
__all__ = [
    'stable_regime_analysis',
    'quick_analysis', 
    'production_regime_analysis',
    'version'
]
