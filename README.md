# 🚀 AutoRegime: AI-Powered Market Regime Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoRegime** is the first **one-line market regime detection library** that automatically identifies Bull, Bear, Crisis, and Goldilocks market conditions using optimized Hidden Markov Models.

## ⚠️ Important Disclaimer

**This tool is for research and analysis purposes only. Past performance does not guarantee future results. This is not financial advice. Consult qualified financial professionals for investment decisions.**

## 🛡️ Reliability Guarantee

```python
# Always get the same result
from autoregime import AutoRegimeDetector
detector = AutoRegimeDetector(random_state=42)  # Deterministic mode

# Or use reliable wrapper
import autoregime
result = autoregime.reliable_regime_analysis('SPY')  # Same result every time
Problem Solved: Eliminated HMM non-deterministic behavior for production use.

✨ Key Features
🤖 One-Line API: autoregime.quick_analysis('SPY') - Complete regime detection
🎯 Human-Readable Results: "Bull Market" vs cryptic numbers (0, 1, 2...)
🔧 Auto-Optimization: Discovers optimal regime count automatically
📊 25+ Years Validated: Tested from 1995 to present
🛡️ Production-Ready: Deterministic, reliable results for live trading
📈 Rich Analytics: Sharpe ratios, drawdowns, confidence scores
🌐 Interactive Dashboard: Real-time regime monitoring
⚡ 30+ Lines → 1 Line: Eliminates complex HMM setup
🚀 Quick Start
Installation

pip install git+https://github.com/KANYINSOLA-OGUNBANJO/-AutoRegime.git
Instant Analysis (2 Lines)
import autoregime
autoregime.quick_analysis('SPY')  # Complete regime analysis instantly
Basic Usage
import autoregime

# Quick demo with real market data
autoregime.quick_demo()

# Launch interactive dashboard  
autoregime.launch_dashboard()

# Reliable analysis (always same result)
result = autoregime.reliable_regime_analysis('SPY')
🔧 Enhanced Stability Features
AutoRegime now includes multiple analysis modes for different use cases:

Stable Analysis (Enhanced Reliability)
import autoregime

# Stable analysis with enhanced parameters
detector = autoregime.stable_regime_analysis('AAPL')
print(f"Stable regimes detected: {detector.optimal_n_regimes}")
Production Analysis (Maximum Reliability)
# Production-ready analysis for live systems
detector = autoregime.production_regime_analysis('SPY')
timeline = detector.get_regime_timeline()
Batch Analysis (Multiple Assets)
# Analyze multiple assets at once
results = autoregime.multi_asset_analysis(['AAPL', 'SPY', 'QQQ'], mode='stable')
for symbol, detector in results.items():
    print(f"{symbol}: {detector.optimal_n_regimes} regimes")
🧪 Model Sensitivity & Reproducibility
AutoRegime exhibits professional-grade behavior:

Same data + same seed = Identical results ✅
Different data periods = Different regimes (expected) ✅
Test Reproducibility
# Verify consistent behavior
autoregime.demonstrate_autoregime_sensitivity()
Why different dates give different results:

Jan 1st vs Jan 3rd = Different market data points
Hidden Markov Models are properly sensitive to data changes
This is expected behavior for professional regime detection
Solution: Use same date ranges for consistent analysis
Stability Modes Comparison
Mode	Max Regimes	Min Duration	Use Case
Standard	6	20 days	Research, exploration
Stable	4	20 days	Balanced analysis
Production	3	30 days	Live trading systems

📊 Demo Codes
Demo 1: Enhanced Stability Analysis
import autoregime

# Stable regime detection
detector = autoregime.stable_regime_analysis('AAPL')
print(f"AAPL Stable Analysis: {detector.optimal_n_regimes} regimes")

# Production-ready analysis  
detector = autoregime.production_regime_analysis('SPY')
print(f"SPY Production Analysis: {detector.optimal_n_regimes} regimes")
Demo 2: Multi-Asset Batch Analysis
# Analyze multiple assets with consistent parameters
results = autoregime.multi_asset_analysis(['AAPL', 'SPY', 'QQQ', 'TLT'], mode='stable')

for symbol, detector in results.items():
    if detector:
        timeline = detector.get_regime_timeline()
        current_regime = timeline.iloc[-1]['Regime_Name']
        print(f"{symbol}: {detector.optimal_n_regimes} regimes - Current: {current_regime}")
Demo 3: Reproducibility Testing
# Demonstrate professional sensitivity behavior
autoregime.demonstrate_autoregime_sensitivity()

# This shows:
# ✅ Same data + same seed = identical results
# ✅ Different data periods = different regimes (expected)
Demo 4: Production Usage
from autoregime import AutoRegimeDetector, MarketDataLoader
import numpy as np

detector = AutoRegimeDetector(random_state=42)  # Guaranteed consistency
loader = MarketDataLoader()
data = loader.load_market_data(['AAPL'], start_date='2020-01-01')
detector.fit(data)
regime = detector.predict_current_regime(data)
print(f"AAPL: {regime}")
🎯 What AutoRegime Detects
🚀 Bull Market: High momentum with controlled volatility
📉 Bear Market: Sustained downward pressure and defensive positioning
🎯 Goldilocks: Ideal conditions - moderate growth with low volatility
📊 Sideways: Range-bound markets requiring balanced strategies
⚠️ Crisis: Extreme volatility requiring immediate risk management
🏆 AutoRegime vs Competitors
Feature	AutoRegime	hmmlearn	Academic Tools
API Complexity	1 line	30+ lines	50+ lines
Result Format	"Bull Market"	0, 1, 2...	0, 1, 2...
Auto-Optimization	✅ Yes	❌ Manual	❌ Manual
Reliability	✅ Deterministic	❌ Random	❌ Random
Production Ready	✅ Yes	❌ Research only	❌ Prototypes
🎯 Real-World Validation
AutoRegime successfully identified major market events:

✅ COVID-19 Crash (March 2020) - Crisis regime detected
✅ 2008 Financial Crisis - Bear market identification
✅ Dot-com Bubble (2000-2002) - Crisis → Bear transition
✅ Bull Runs (2009-2020, 2020-2021) - Sustained bull detection
✅ Current Market (Sept 2025) - "Steady Growth" regime
📈 Example Output
Standard Analysis
============================================================
AUTOREGIME ANALYSIS SUMMARY
============================================================
Optimal number of regimes: 4

Bull Market:
  Frequency: 30.5%
  Annual Return: 38.9%
  Sharpe Ratio: 0.83
  
Goldilocks:  
  Frequency: 69.5%
  Annual Return: 20.2%
  Sharpe Ratio: 0.86

CURRENT MARKET STATUS:
   Active Regime: Bull Market
   Confidence Level: 100.0%
   Expected Return: 38.9% annually
   Strategy: INCREASE EQUITY ALLOCATION
Stability Mode Output
🔧 Stable Regime Analysis for AAPL
Enhanced stability parameters active...

============================================================
AUTOREGIME ANALYSIS SUMMARY
🔧 STABILITY MODE ACTIVE
============================================================
Optimal number of regimes: 3

Goldilocks (Regime 0):
  Frequency: 85.2%
  Avg Duration: 45.3 days
  Annual Return: 28.4%
  Sharpe Ratio: 1.24

Bull Market (Regime 1):  
  Frequency: 12.1%
  Avg Duration: 32.1 days
  Annual Return: 52.7%
  Sharpe Ratio: 1.89

Risk-Off (Regime 2):
  Frequency: 2.7%
  Avg Duration: 28.0 days  
  Annual Return: -15.3%
  Sharpe Ratio: -0.45

✅ Stable Analysis Complete for AAPL
📊 Detected 3 stable regimes
🛠 Requirements
Python 3.8+
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
hmmlearn >= 0.2.7
matplotlib >= 3.5.0
plotly >= 5.0.0
streamlit >= 1.28.0
yfinance >= 0.1.87
🔧 Installation Issues?
Common Fixes:

# Development install
git clone https://github.com/KANYINSOLA-OGUNBANJO/-AutoRegime.git
cd -AutoRegime  
pip install -e .

# Update dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
🤝 Contributing
Contributions welcome! Areas for improvement:

Additional regime types (Momentum, Reversal)
More asset classes (Crypto, Bonds, Commodities)
Enhanced visualization features
Performance optimizations
Please read CONTRIBUTING.md for guidelines.

🌟 Why AutoRegime?
The Problem: Existing regime detection requires:

30+ lines of complex HMM setup
Manual parameter tuning
Deep ML expertise
Produces cryptic, unreliable results
AutoRegime Solution:

✅ One-line API: autoregime.quick_analysis('SPY')
✅ Human-readable regimes: "Bull Market" not "State 2"
✅ Auto-optimization: No manual tuning needed
✅ Production reliability: Consistent results every time
✅ Rich insights: Sharpe ratios, confidence scores, recommendations
📊 Business Applications
Portfolio Management: Dynamic allocation based on regime
Risk Management: Early warning system for regime changes
Systematic Trading: Regime-aware strategy adjustments
Research: Academic and institutional market analysis
📝 License
MIT License - see LICENSE file for details.

🏅 About the Author
Built by Kanyinsola Ogunbanjo - Quantitative Finance Professional

📧 Email: kanyinsolaogunbanjo@gmail.com
🔗 LinkedIn: Kanyinsola Ogunbanjo
🐙 GitHub: @KANYINSOLA-OGUNBANJO
⭐ Star this repo if AutoRegime helps your market analysis!

🍴 Fork it to contribute your own regime detection improvements!

📈 Try it now: pip install git+https://github.com/KANYINSOLA-OGUNBANJO/-AutoRegime.git

