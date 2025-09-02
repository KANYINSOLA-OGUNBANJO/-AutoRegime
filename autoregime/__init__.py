"""
AutoRegime: AI-Powered Market Regime Detection
One-line installation and usage for instant market analysis
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Core imports for easy access
from .core.regime_detection import AutoRegimeDetector
from .utils.data_loader import MarketDataLoader
from .visualization.regime_plots import RegimeVisualizer

def quick_demo():
    """
    Run complete AutoRegime demo with one function call.
    """
    print("🚀 AutoRegime Quick Demo Starting...")
    print("="*50)
    
    try:
        # Load real market data
        print("📊 Loading real market data (2022-2024)...")
        loader = MarketDataLoader()
        data = loader.load_preset_universe('indices', start_date='2022-01-01')
        
        print(f"Loaded {data.shape[0]} days of data for {data.shape[1]} assets")
        print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Run AutoRegime detection
        print("\n🧠 Running AI regime detection...")
        detector = AutoRegimeDetector(max_regimes=4, verbose=True)
        detector.fit(data)
        
        # Current regime prediction
        print("\n🎯 CURRENT MARKET ANALYSIS")
        print("-"*30)
        
        recent_data = data.tail(21)
        current_regime, confidence = detector.predict_current_regime(recent_data)
        regime_name = detector.regime_names.get(current_regime, f'Regime {current_regime}')
        
        print(f"CURRENT MARKET REGIME: {regime_name}")
        print(f"Confidence Level: {confidence:.1%}")
        
        # Show regime characteristics
        if current_regime in detector.regime_characteristics:
            char = detector.regime_characteristics[current_regime]
            print(f"\n📋 {regime_name} Characteristics:")
            print(f"  💰 Expected Annual Return: {char['mean_return']:.1%}")
            print(f"  📊 Expected Volatility: {char['volatility']:.1%}")
            print(f"  🎯 Sharpe Ratio: {char['sharpe_ratio']:.2f}")
        
        # Create visualizations
        print(f"\n🎨 Creating visualizations...")
        visualizer = RegimeVisualizer(detector, data)
        timeline_fig = visualizer.plot_regime_timeline(interactive=False)
        
        print(f"\n🎉 AutoRegime Demo Complete!")
        print("="*50)
        
        return detector, data, visualizer
        
    except Exception as e:
        print(f"Error in demo: {str(e)}")
        return None, None, None

def launch_dashboard():
    """
    Launch interactive AutoRegime dashboard.
    """
    import subprocess
    import sys
    import os
    
    print("🌐 Launching AutoRegime Dashboard...")
    
    try:
        # Find dashboard file
        dashboard_path = os.path.join(os.path.dirname(__file__), 'visualization', 'dashboard.py')
        
        if not os.path.exists(dashboard_path):
            print("Dashboard file not found!")
            print("Try creating a simple dashboard file or use the demo instead")
            print("Run: quick_demo() for visualizations")
            return
        
        print("Starting Streamlit server...")
        print("Dashboard will open in your browser automatically")
        print("URL: http://localhost:8501")
        print("\n💡 To stop the dashboard: Press Ctrl+C in terminal")
        
        # Launch Streamlit
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', dashboard_path])
        
    except Exception as e:
        print(f"Error launching dashboard: {str(e)}")
        print("\n💡 Alternative: Try the demo with visualizations")
        print("Run: quick_demo()")

def quick_analysis(symbols, start_date='2023-01-01'):
    """Quick regime analysis for custom assets."""
    print(f"Quick Analysis: {symbols}")
    print("-"*30)
    
    try:
        loader = MarketDataLoader()
        if isinstance(symbols, str):
            symbols = [symbols]
        
        data = loader.load_market_data(symbols, start_date=start_date)
        detector = AutoRegimeDetector(max_regimes=4, verbose=False)
        detector.fit(data)
        
        print(f"Discovered {detector.optimal_n_regimes} regimes")
        
        current_regime, confidence = detector.predict_current_regime(data.tail(21))
        regime_name = detector.regime_names.get(current_regime, f'Regime {current_regime}')
        print(f"Current regime: {regime_name} ({confidence:.1%})")
        
        return detector, data
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return None, None

def version():
    """Display AutoRegime version and system info."""
    import sys
    print("AutoRegime System Information")
    print("="*40)
    print(f"AutoRegime Version: {__version__}")
    print(f"Python Version: {sys.version.split()[0]}")
    print("="*40)

def full_historical_analysis():
    """
    Analyze MAXIMUM historical data available.
    
    This will discover regimes across multiple market cycles:
    - Dot-com bubble (1999-2002)
    - Early 2000s bear market
    - Mid-2000s bull run  
    - 2008 Financial Crisis
    - 2009-2020 Bull market
    - COVID crash & recovery
    - Current period
    """
    print("🕰️ FULL HISTORICAL AUTOREGIME ANALYSIS")
    print("="*60)
    print("Analyzing 24+ years of market data across multiple cycles...")
    
    try:
        loader = MarketDataLoader()
        
        # Load maximum historical data
        print("📊 Loading maximum historical data...")
        data = loader.load_market_data(['SPY', 'QQQ', 'DIA'], start_date='2000-01-01')
        
        years = len(data) / 252
        print(f"Loaded {years:.1f} years of data ({len(data)} trading days)")
        print(f"Full period: {data.index[0].date()} to {data.index[-1].date()}")
        
        print("\n🔍 Expected regime discoveries:")
        print("  • Dot-com crash (2000-2002)")
        print("  • Early 2000s recovery (2003-2007)")  
        print("  • 2008 Financial Crisis")
        print("  • Post-crisis recovery (2009-2020)")
        print("  • COVID crash & recovery (2020-2021)")
        print("  • Current period (2022-2024)")
        
        # Run comprehensive regime detection
        print(f"\n🧠 Running regime detection on {years:.1f} years of data...")
        print("This may take 2-3 minutes due to large dataset...")
        
        detector = AutoRegimeDetector(
            max_regimes=8,      # Allow up to 8 regimes for long history
            min_regime_duration=10,  # Longer minimum for stability
            verbose=True
        )
        
        detector.fit(data)
        
        # Analyze results
        print(f"\n🎯 HISTORICAL REGIME ANALYSIS RESULTS:")
        print(f"Discovered {detector.optimal_n_regimes} distinct regimes across {years:.1f} years")
        
        # Show regime characteristics
        for regime_id, char in detector.regime_characteristics.items():
            regime_name = detector.regime_names.get(regime_id, f'Regime {regime_id}')
            print(f"\n{regime_name}:")
            print(f"  Frequency: {char['frequency']:.1%}")
            print(f"  Avg Duration: {char['avg_duration']:.1f} days")
            print(f"  Annual Return: {char['mean_return']:.1%}")
            print(f"  Annual Volatility: {char['volatility']:.1%}")
            print(f"  Sharpe Ratio: {char['sharpe_ratio']:.2f}")
        
        # Create historical timeline
        print(f"\n🎨 Creating historical regime timeline...")
        visualizer = RegimeVisualizer(detector, data)
        timeline_fig = visualizer.plot_regime_timeline(interactive=False)
        
        # Current regime
        current_regime, confidence = detector.predict_current_regime(data.tail(21))
        regime_name = detector.regime_names.get(current_regime, f'Regime {current_regime}')
        print(f"\n🚨 CURRENT REGIME (based on 24+ year analysis): {regime_name}")
        print(f"📊 Confidence: {confidence:.1%}")
        
        print(f"\n🎉 Full historical analysis completed!")
        print(f"Successfully analyzed {years:.1f} years across multiple market cycles!")
        
        return detector, data, visualizer
        
    except Exception as e:
        print(f"Error in historical analysis: {e}")
        return None, None, None
# Make everything easily accessible
__all__ = [
    'AutoRegimeDetector',
    'MarketDataLoader', 
    'RegimeVisualizer',
    'quick_demo',
    'launch_dashboard',
    'quick_analysis',
    'full_historical_analysis',  # Add this line
    'version'
]


print("AutoRegime loaded! Try: quick_demo() or launch_dashboard()")

