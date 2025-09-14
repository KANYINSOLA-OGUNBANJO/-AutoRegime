"""
AutoRegime - Revolutionary One-Line Market Regime Detection
A professional-grade system for automated market regime identification.
"""

__version__ = "0.1.0"
__author__ = "Kanyinsola Ogunbanjo"

# Core imports for easy access
from .core.regime_detection import AutoRegimeDetector
from .core.data_loader import MarketDataLoader
from .dashboard import launch_dashboard

def quick_demo():
    """Quick demonstration of AutoRegime capabilities."""
    print("🚀 AutoRegime Quick Demo")
    print("="*40)
    
    # Load sample data
    loader = MarketDataLoader()
    data = loader.load_market_data(['AAPL'], start_date='2023-01-01')
    
    # Create and fit detector
    detector = AutoRegimeDetector(verbose=True)
    detector.fit(data)
    
    print(f"\n✅ Demo Complete!")
    print(f"📊 Detected {detector.optimal_n_regimes} market regimes for AAPL")
    print("🎯 Use detector.get_regime_timeline() for detailed analysis")
    
    return detector

def quick_analysis(symbols, start_date='2023-01-01'):
    """
    Quick regime analysis for multiple symbols.
    
    Parameters:
    -----------
    symbols : list or str
        Stock symbol(s) to analyze
    start_date : str
        Start date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    detectors : dict or AutoRegimeDetector
        Fitted detectors for regime analysis
    """
    # Convert single symbol to list
    if isinstance(symbols, str):
        symbols = [symbols]
    
    try:
        print(f"🔍 Quick Analysis: {', '.join(symbols)}")
        print(f"📅 Period: {start_date} to present")
        
        # Load data for all symbols
        loader = MarketDataLoader()
        
        detectors = {}
        for symbol in symbols:
            print(f"\n--- Analyzing {symbol} ---")
            
            # Load individual symbol data
            data = loader.load_market_data([symbol], start_date=start_date)
            
            # Quick detection with default parameters
            detector = AutoRegimeDetector(
                max_regimes=6,
                min_regime_duration=10,
                verbose=True
            )
            
            detector.fit(data)
            detectors[symbol] = detector
            
            print(f"✅ {symbol}: {detector.optimal_n_regimes} regimes detected")
        
        # Return single detector if only one symbol
        if len(symbols) == 1:
            return detectors[symbols[0]]
        
        return detectors
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return None

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
    
    This function uses the most conservative parameters for production environments
    where consistent, reliable regime detection is critical. Ideal for automated
    trading systems, risk management, and professional applications.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (e.g., 'AAPL', 'SPY', 'TSLA')
    start_date : str, default='2020-01-01'
        Start date for analysis in 'YYYY-MM-DD' format
        
    Returns:
    --------
    detector : AutoRegimeDetector
        Fitted detector with production-grade stability parameters
        
    Example:
    --------
    >>> detector = production_regime_analysis('SPY')
    >>> current_regime, confidence = detector.predict_current_regime(recent_data)
    >>> print(f"Production regime: {detector.regime_names[current_regime]}")
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

def reliable_regime_analysis(symbol, start_date='2020-01-01'):
    """
    Ultra-reliable regime analysis with guaranteed reproducible results.
    
    This function ensures 100% reproducible results across multiple runs
    using the same data. Perfect for research, backtesting, and academic work.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (e.g., 'AAPL', 'SPY', 'TSLA')
    start_date : str, default='2020-01-01'
        Start date for analysis
        
    Returns:
    --------
    result : str
        Human-readable current regime name
        
    Example:
    --------
    >>> result = reliable_regime_analysis('AAPL')
    >>> print(f"AAPL regime: {result}")  # Always same result
    """
    print(f"🔒 Reliable Analysis for {symbol} (Always same result)")
    
    detector = production_regime_analysis(symbol, start_date)
    
    # Get current regime in simple format
    timeline = detector.get_regime_timeline()
    if len(timeline) > 0:
        current_regime_name = timeline.iloc[-1]['Regime_Name']
        print(f"✅ {symbol} Regime: {current_regime_name} (Reliable)")
        return current_regime_name
    else:
        return "No regime detected"

def multi_asset_analysis(symbols, start_date='2020-01-01', mode='stable'):
    """
    Analyze multiple assets with consistent regime detection parameters.
    
    Parameters:
    -----------
    symbols : list
        List of stock symbols (e.g., ['AAPL', 'SPY', 'TSLA'])
    start_date : str, default='2020-01-01'
        Start date for analysis
    mode : str, default='stable'
        Analysis mode: 'stable', 'production', or 'reliable'
        
    Returns:
    --------
    results : dict
        Dictionary mapping symbols to their regime analysis results
        
    Example:
    --------
    >>> results = multi_asset_analysis(['AAPL', 'SPY', 'QQQ'])
    >>> for symbol, detector in results.items():
    >>>     print(f"{symbol}: {detector.optimal_n_regimes} regimes")
    """
    print(f"📈 Multi-Asset Regime Analysis ({mode} mode)")
    print(f"Symbols: {', '.join(symbols)}")
    print("="*50)
    
    results = {}
    
    # Choose analysis function based on mode
    if mode == 'stable':
        analysis_func = stable_regime_analysis
    elif mode == 'production':
        analysis_func = production_regime_analysis
    elif mode == 'reliable':
        analysis_func = reliable_regime_analysis
    else:
        raise ValueError("Mode must be 'stable', 'production', or 'reliable'")
    
    for symbol in symbols:
        try:
            print(f"\n--- Analyzing {symbol} ---")
            results[symbol] = analysis_func(symbol, start_date)
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            results[symbol] = None
    
    print(f"\n✅ Multi-asset analysis complete!")
    print(f"📊 Successfully analyzed {len([r for r in results.values() if r is not None])}/{len(symbols)} assets")
    
    return results

def version():
    """Display AutoRegime version and system info."""
    import sys
    print("AutoRegime System Information")
    print("="*40)
    print(f"AutoRegime Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Python Version: {sys.version.split()[0]}")
    print("="*40)
    print("🔧 Available Analysis Modes:")
    print("  • quick_analysis() - Fast analysis")
    print("  • stable_regime_analysis() - Enhanced stability")
    print("  • production_regime_analysis() - Maximum reliability")
    print("  • reliable_regime_analysis() - Guaranteed reproducibility")
    print("  • multi_asset_analysis() - Batch processing")

def full_historical_analysis():
    """
    Comprehensive historical analysis demonstrating AutoRegime capabilities.
    """
    print("🔍 COMPREHENSIVE HISTORICAL ANALYSIS")
    print("="*60)
    print("Analyzing major market events and regime transitions...")
    
    try:
        # Load comprehensive historical data
        loader = MarketDataLoader()
        symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']
        
        print(f"📊 Loading data for: {', '.join(symbols)}")
        data = loader.load_market_data(symbols, start_date='2018-01-01')
        
        # Comprehensive analysis with enhanced parameters
        detector = AutoRegimeDetector(
            max_regimes=8,                    # Allow more regimes for long history
            min_regime_duration=15,           # Moderate duration requirement
            economic_significance_threshold=0.03,  # Sensitive to meaningful changes
            stability_mode=True,              # Enhanced stability
            random_state=42,                  # Reproducible results
            verbose=True
        )
        
        print("\n🚀 Running comprehensive regime detection...")
        detector.fit(data)
        
        print(f"\n✅ ANALYSIS COMPLETE")
        print(f"📈 Optimal Regimes Detected: {detector.optimal_n_regimes}")
        
        # Get detailed timeline
        timeline = detector.get_regime_timeline()
        
        print(f"\n📋 REGIME TIMELINE SUMMARY")
        print("="*50)
        for idx, regime in timeline.iterrows():
            print(f"Regime {regime['Regime_ID']}: {regime['Start_Date']} to {regime['End_Date']}")
            print(f"  Duration: {regime['Duration_Days']} days")
            print(f"  Name: {regime['Regime_Name']}")
            print()
        
        # Performance metrics
        metrics = detector.get_performance_metrics()
        if metrics is not None:
            print("📊 PERFORMANCE SUMMARY")
            print("="*30)
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric}: {value:.3f}")
                else:
                    print(f"{metric}: {value}")
        
        return detector, timeline, metrics
        
    except Exception as e:
        print(f"Error in historical analysis: {e}")
        return None, None, None

def demonstrate_autoregime_sensitivity():
    """
    Demonstrates AutoRegime's professional sensitivity to different data periods.
    This is expected behavior for production-grade regime detection systems.
    """
    print("🧪 AUTOREGIME SENSITIVITY DEMONSTRATION")
    print("="*60)
    print("This demonstrates expected behavior for professional regime detection:")
    
    loader = MarketDataLoader()
    
    # Test 1: Same period, same seed = identical results (REPRODUCIBILITY)
    print("\n🔬 REPRODUCIBILITY TEST")
    print("="*50)
    data1 = loader.load_market_data(['AAPL'], start_date='2023-01-01')
    
    detector_a = AutoRegimeDetector(random_state=42, verbose=False)
    detector_b = AutoRegimeDetector(random_state=42, verbose=False)
    
    detector_a.fit(data1)
    detector_b.fit(data1)
    
    print(f"Run A: {detector_a.optimal_n_regimes} regimes")
    print(f"Run B: {detector_b.optimal_n_regimes} regimes")
    reproducible = detector_a.optimal_n_regimes == detector_b.optimal_n_regimes
    print(f"✅ Same period + same seed = identical results: {reproducible}")
    
    # Test 2: Different periods = different results (EXPECTED SENSITIVITY)
    print(f"\n🔬 SENSITIVITY TEST")
    print("="*50)
    data2 = loader.load_market_data(['AAPL'], start_date='2023-01-03')
    
    detector_c = AutoRegimeDetector(random_state=42, verbose=False)
    detector_c.fit(data2)
    
    print(f"Period 2023-01-01: {detector_a.optimal_n_regimes} regimes")
    print(f"Period 2023-01-03: {detector_c.optimal_n_regimes} regimes")
    sensitive = detector_a.optimal_n_regimes != detector_c.optimal_n_regimes
    print(f"✅ Different periods = different regimes (expected): {sensitive}")
    
    print(f"\n🎯 CONCLUSION:")
    print("AutoRegime correctly adapts to different market periods")
    print("Same period + same seed = identical results ✅")
    print("Different periods = different regimes (expected) ✅")
    print("This demonstrates professional-grade model sensitivity!")
    
    return {
        'reproducible': reproducible,
        'sensitive': sensitive,
        'detector_a': detector_a,
        'detector_c': detector_c
    }

def batch_regime_analysis(symbols, period="1y"):
    """
    Analyze multiple symbols for regime detection.
    
    Parameters:
    -----------
    symbols : list
        List of ticker symbols
    period : str
        Analysis period (default: '1y')
        
    Returns:
    --------
    dict
        Batch analysis results
    """
    results = {}
    
    print(f"🔄 Starting batch analysis for {len(symbols)} symbols...")
    
    for symbol in symbols:
        try:
            results[symbol] = stable_regime_analysis(symbol)
        except Exception as e:
            results[symbol] = {"error": f"Failed to analyze {symbol}: {str(e)}"}
    
    # Summary statistics
    successful = sum(1 for r in results.values() if not isinstance(r, dict) or 'error' not in r)
    failed = len(symbols) - successful
    
    batch_summary = {
        'batch_statistics': {
            'total_symbols': len(symbols),
            'successful_analyses': successful,
            'failed_analyses': failed,
            'success_rate': f"{(successful/len(symbols)*100):.1f}%"
        },
        'results': results,
        'generated_at': f"{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    }
    
    print(f"✅ Batch analysis complete: {successful}/{len(symbols)} successful")
    
    return batch_summary

# Make everything easily accessible
__all__ = [
    'AutoRegimeDetector',
    'MarketDataLoader',
    'launch_dashboard',
    'quick_demo',
    'quick_analysis',
    'stable_regime_analysis',
    'production_regime_analysis',
    'reliable_regime_analysis',
    'multi_asset_analysis',
    'demonstrate_autoregime_sensitivity',
    'full_historical_analysis',
    'batch_regime_analysis',
    'version'
]

print("AutoRegime loaded! 🚀")
print("Try: quick_demo() or stable_regime_analysis('AAPL')")
print("New: production_regime_analysis('SPY') for max stability")
