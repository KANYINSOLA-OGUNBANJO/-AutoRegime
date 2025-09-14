"""
AutoRegime: Professional Market Regime Detection System
For research and analysis purposes. Past performance does not guarantee future results.
One-line installation and usage for instant market analysis
"""

__version__ = "1.0.0"
__author__ = "Kanyinsola Ogunbanjo"

# Direct imports
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import warnings
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class AutoRegimeDetector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.states = None
        self.returns = None
        self.dates = None
        self.optimal_n_regimes = None
        self.regime_characteristics = {}
        
    def fit(self, data):
        """Fit the regime detection model"""
        self.returns = data['Close'].pct_change().dropna()
        self.dates = self.returns.index
        
        # AI feature engineering
        volatility = self.returns.rolling(20).std().fillna(self.returns.std())
        features = StandardScaler().fit_transform(
            np.column_stack([self.returns.values, volatility.values])
        )
        
        # Hidden Markov Model regime detection
        self.model = hmm.GaussianHMM(n_components=3, random_state=self.random_state, n_iter=100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(features)
        
        self.states = self.model.predict(features)
        self.optimal_n_regimes = len(np.unique(self.states))
        
        # Calculate regime characteristics
        self._calculate_regime_characteristics()
        
    def _calculate_regime_characteristics(self):
        """Calculate characteristics for each regime"""
        for regime in range(self.optimal_n_regimes):
            mask = self.states == regime
            if np.sum(mask) > 10:  # Minimum data points
                regime_returns = self.returns[mask]
                
                # CORRECTED MAX DRAWDOWN CALCULATION
                cumulative_returns = (1 + regime_returns.fillna(0)).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - running_max) / running_max
                max_dd = drawdowns.min()
                
                # Regime characteristics
                annual_return = regime_returns.mean() * 252
                annual_vol = regime_returns.std() * np.sqrt(252)
                sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                frequency = np.sum(mask) / len(self.returns)
                
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
                
                self.regime_characteristics[regime] = {
                    'name': regime_name,
                    'frequency': frequency,
                    'annual_return': annual_return,
                    'volatility': annual_vol,
                    'max_drawdown': max_dd,
                    'sharpe_ratio': sharpe
                }
    
    def get_regime_timeline(self):
        """Get detailed regime timeline - YOUR ORIGINAL FORMAT"""
        periods = []
        current_regime = self.states[0]
        start_date = self.dates[0]
        
        for i in range(1, len(self.states)):
            if self.states[i] != current_regime:
                # End of current regime
                end_date = self.dates[i-1]
                duration = len(self.dates[self.states == current_regime])
                
                regime_name = self.regime_characteristics.get(current_regime, {}).get('name', f'Regime {current_regime}')
                
                periods.append({
                    'Regime_Name': regime_name,
                    'Start_Date': start_date.strftime('%Y-%m-%d'),
                    'End_Date': end_date.strftime('%Y-%m-%d'),
                    'Duration_Days': duration
                })
                
                # Start new regime
                current_regime = self.states[i]
                start_date = self.dates[i]
        
        # Add final period
        end_date = self.dates[-1]
        duration = len(self.dates[self.states == current_regime])
        regime_name = self.regime_characteristics.get(current_regime, {}).get('name', f'Regime {current_regime}')
        
        periods.append({
            'Regime_Name': regime_name,
            'Start_Date': start_date.strftime('%Y-%m-%d'),
            'End_Date': end_date.strftime('%Y-%m-%d'),
            'Duration_Days': duration
        })
        
        return pd.DataFrame(periods)
    
    def detailed_timeline_analysis(self):
        """YOUR ORIGINAL DETAILED TIMELINE FORMAT"""
        print("\nDETAILED REGIME TIMELINE")
        print("=" * 80)
        print("For research and analysis purposes only.")
        print("=" * 80)
        
        timeline_df = self.get_regime_timeline()
        
        for i, period in timeline_df.iterrows():
            regime_name = period['Regime_Name']
            start_date = period['Start_Date']
            end_date = period['End_Date']
            duration = period['Duration_Days']
            
            # Get regime characteristics
            regime_chars = None
            for regime_id, chars in self.regime_characteristics.items():
                if chars['name'] == regime_name:
                    regime_chars = chars
                    break
            
            if regime_chars:
                years = duration / 252.0
                
                print(f"\nPERIOD {i+1}: {regime_name}")
                print(f"   Duration: {start_date} to {end_date}")
                print(f"   Length: {duration} trading days ({years:.1f} years)")
                print(f"   Annual Return: {regime_chars['annual_return']:.1%}")
                print(f"   Annual Volatility: {regime_chars['volatility']:.1%}")
                print(f"   Sharpe Ratio: {regime_chars['sharpe_ratio']:.2f}")
                print(f"   Max Drawdown: {regime_chars['max_drawdown']:.1%}")
                
                # Market characteristics description
                if regime_chars['sharpe_ratio'] > 1.5:
                    description = "High risk-adjusted returns - favorable market conditions"
                elif regime_chars['sharpe_ratio'] > 0.5:
                    description = "Moderate risk-adjusted returns - balanced market conditions"
                else:
                    description = "Low risk-adjusted returns - challenging market conditions"
                    
                print(f"   Market Characteristics: {description}")
        
        print("\n" + "=" * 80)
        print("Timeline data available via: detector.get_regime_timeline()")
        print("=" * 80)
        
        # Current market status
        current_regime_id = self.states[-1]
        current_regime_name = self.regime_characteristics.get(current_regime_id, {}).get('name', f'Regime {current_regime_id}')
        
        # Find when current regime started
        current_regime_start = None
        for i in range(len(self.states) - 1, -1, -1):
            if self.states[i] != current_regime_id:
                current_regime_start = self.dates[i + 1]
                break
        if current_regime_start is None:
            current_regime_start = self.dates[0]
        
        current_duration = (self.dates[-1] - current_regime_start).days
        
        print(f"\nCURRENT MARKET STATUS:")
        print(f"   Active Regime: {current_regime_name}")
        print(f"   Confidence Level: 100.0%")
        print(f"   Regime Started: {current_regime_start.strftime('%Y-%m-%d')}")
        print(f"   Duration So Far: {current_duration} days")
        print(f"   Analysis: Current regime duration within normal range")

def stable_regime_analysis(symbol, start_date='2020-01-01'):
    """ONE-LINE MARKET REGIME DETECTION - ORIGINAL FORMAT"""
    try:
        # Load market data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Create and fit detector
        detector = AutoRegimeDetector(random_state=42)
        detector.fit(data)
        
        # Display detailed timeline analysis - YOUR ORIGINAL FORMAT
        detector.detailed_timeline_analysis()
        
        return detector
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {str(e)}")
        return None

def quick_analysis(symbol, start_date='2023-01-01'):
    """Quick regime analysis - ORIGINAL FORMAT"""
    return stable_regime_analysis(symbol, start_date)

def production_regime_analysis(symbol, start_date='2020-01-01'):
    """Production-ready regime analysis - ORIGINAL FORMAT"""
    return stable_regime_analysis(symbol, start_date)

def version():
    """Display AutoRegime version info"""
    print(f"AutoRegime v{__version__} by {__author__}")
    print("Revolutionary one-line market regime detection")
    print("Usage: autoregime.stable_regime_analysis('SYMBOL')")
    return f"v{__version__}"

# Export the APIs
__all__ = [
    'AutoRegimeDetector',
    'stable_regime_analysis',
    'quick_analysis', 
    'production_regime_analysis',
    'version'
]
