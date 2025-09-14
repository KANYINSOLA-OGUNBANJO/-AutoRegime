"""
AutoRegime: Professional Market Regime Detection System
For research and analysis purposes. Past performance does not guarantee future results.
One-line installation and usage for instant market analysis
"""

__version__ = "0.1.0"
__author__ = "Kanyinsola Ogunbanjo"

# Core imports for easy access
from .core.regime_detection import AutoRegimeDetector
from .utils.data_loader import MarketDataLoader
# REMOVED: from .visualization.regime_plots import RegimeVisualizer  # <-- REMOVE THIS LINE TO FIX NUMPY ERROR

def quick_demo():
    """
    Run complete AutoRegime demo with one function call.
    """
    print("AutoRegime Quick Demo Starting...")
    print("="*50)
    
    try:
        # Load real market data
        print("Loading real market data (2022-2024)...")
        loader = MarketDataLoader()
        data = loader.load_preset_universe('indices', start_date='2022-01-01')
        
        print(f"Loaded {data.shape[0]} days of data for {data.shape[1]} assets")
        print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Run AutoRegime detection
        print("\nRunning AI regime detection...")
        detector = AutoRegimeDetector(max_regimes=4, verbose=True)
        detector.fit(data)
        
        # Current regime prediction
        print("\nCURRENT MARKET ANALYSIS")
        print("-"*30)
        
        recent_data = data.tail(21)
        current_regime, confidence = detector.predict_current_regime(recent_data)
        regime_name = detector.regime_names.get(current_regime, f'Regime {current_regime}')
        
        print(f"CURRENT MARKET REGIME: {regime_name}")
        print(f"Confidence Level: {confidence:.1%}")
        
        # Show regime characteristics
        if current_regime in detector.regime_characteristics:
            char = detector.regime_characteristics[current_regime]
            print(f"\n{regime_name} Characteristics:")
            print(f"  Expected Annual Return: {char['mean_return']:.1%}")
            print(f"  Expected Volatility: {char['volatility']:.1%}")
            print(f"  Sharpe Ratio: {char['sharpe_ratio']:.2f}")
        
        print(f"\nAutoRegime Demo Complete!")
        print("="*50)
        
        return detector, data
        
    except Exception as e:
        print(f"Error in demo: {str(e)}")
        return None, None

def stable_regime_analysis(symbol, start_date='2020-01-01'):
    """
    Perform stable regime analysis with enhanced parameters for consistent results.
    
    This function provides more robust regime detection by using stability-enhanced
    parameters that reduce noise and produce longer-duration, more meaningful regimes.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (e.g., 'AAPL', 'SPY', 'TSLA')
    start_date : str, default='2020-01-01'
        Start date for analysis in 'YYYY-MM-DD' format
        
    Returns:
    --------
    detector : AutoRegimeDetector
        Fitted detector with stable parameters and regime analysis results
        
    Example:
    --------
    >>> detector = stable_regime_analysis('AAPL')
    >>> timeline = detector.get_regime_timeline()
    >>> print(f"Detected {detector.optimal_n_regimes} stable regimes")
    """
    print(f"🔧 Stable Regime Analysis for {symbol}")
    print("Enhanced stability parameters active...")
    
    loader = MarketDataLoader()
    data = loader.load_market_data([symbol], start_date=start_date)
    
    # Stability-enhanced parameters
    detector = AutoRegimeDetector(
        stability_mode=True,         # Enable stability mode
        random_state=42,             # Deterministic results
        min_regime_duration=20,      # Longer minimum duration
        max_regimes=4,               # Limit complexity  
        economic_significance_threshold=0.05,  # Higher threshold for significance
        verbose=True
    )
    
    detector.fit(data)
    
    print(f"\n✅ Stable Analysis Complete for {symbol}")
    print(f"📊 Detected {detector.optimal_n_regimes} stable regimes")
    print("🎯 Use detector.get_regime_timeline() for detailed timeline")
    
    return detector

def production_regime_analysis(symbol, start_date='2020-01-01'):
    """
    Production-ready regime analysis optimized for maximum stability and reliability.
    """
    print(f"🏭 Production Regime Analysis for {symbol}")
    print("Maximum stability parameters active...")
    
    loader = MarketDataLoader()
    data = loader.load_market_data([symbol], start_date=start_date)
    
    # Production-grade parameters (maximum stability)
    detector = AutoRegimeDetector(
        stability_mode=True,         # Enable stability mode
        random_state=42,             # Deterministic results
        min_regime_duration=30,      # Very stable regimes (30+ days)
        max_regimes=3,               # Simple, interpretable (max 3 regimes)
        economic_significance_threshold=0.08,  # High significance threshold (8%)
        verbose=False                # Clean output for production
    )
    
    detector.fit(data)
    
    print(f"📊 Production Analysis Complete for {symbol}")
    print(f"🎯 Regimes Detected: {detector.optimal_n_regimes}")
    print(f"⚡ Model ready for production use")
    
    return detector

def quick_analysis(symbol, start_date='2023-01-01'):
    """Quick regime analysis for custom assets."""
    print(f"Quick Analysis: {symbol}")
    print("-"*30)
    
    try:
        loader = MarketDataLoader()
        data = loader.load_market_data([symbol], start_date=start_date)
        detector = AutoRegimeDetector(max_regimes=4, verbose=False)
        detector.fit(data)
        
        print(f"Discovered {detector.optimal_n_regimes} regimes")
        
        current_regime, confidence = detector.predict_current_regime(data.tail(21))
        regime_name = detector.regime_names.get(current_regime, f'Regime {current_regime}')
        print(f"Current regime: {regime_name} ({confidence:.1%})")
        
        return detector
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return None

# Make everything easily accessible
__all__ = [
    'AutoRegimeDetector',
    'MarketDataLoader', 
    'quick_demo',
    'quick_analysis',
    'stable_regime_analysis',
    'production_regime_analysis',
    'version'
]
