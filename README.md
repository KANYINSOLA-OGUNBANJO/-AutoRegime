# 🚀 AutoRegime: AI-Powered Market Regime Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoRegime** is the world's first **one-line market regime detection library** that automatically identifies Bull, Bear, Crisis, and Goldilocks market conditions using optimized Hidden Markov Models.

## ⚠️ Important Disclaimer

**This tool is for research and analysis purposes only. Past performance does not guarantee future results. This is not financial advice. Consult qualified financial professionals for investment decisions.**

## 🛡️ Reliability Guarantee

```python
# Always get the same result
from autoregime import AutoRegimeDetector
detector = AutoRegimeDetector(random_state=42)  # Deterministic mode

# Or use reliable wrapper
from reliable_regime import reliable_regime_analysis
result = reliable_regime_analysis('SPY')  # Same result every time
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
Copypip install git+https://github.com/KANYINSOLA-OGUNBANJO/AutoRegime.git
Instant Analysis (2 Lines)
Copyimport autoregime
autoregime.quick_analysis('SPY')  # Complete regime analysis instantly
Basic Usage
Copyimport autoregime

# Quick demo with real market data
autoregime.quick_demo()

# Launch interactive dashboard  
autoregime.launch_dashboard()

# Reliable analysis (always same result)
result = autoregime.reliable_quick_analysis('SPY')
📊 Demo Codes
Demo 1: Basic Detection
Copyfrom reliable_regime import reliable_regime_analysis
result = reliable_regime_analysis('SPY')
print(f"SPY Regime: {result} (Always consistent)")
Demo 2: Multi-Asset Analysis
Copyfrom reliable_regime import reliable_multi_asset
results = reliable_multi_asset(['SPY', 'QQQ', 'TLT'])
for symbol, result in results.items():
    print(f"{symbol}: {result}")
Demo 3: Production Usage
Copyfrom autoregime import AutoRegimeDetector, MarketDataLoader
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
Copy# Development install
git clone https://github.com/KANYINSOLA-OGUNBANJO/AutoRegime.git
cd AutoRegime  
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

📈 Try it now: pip install git+https://github.com/KANYINSOLA-OGUNBANJO/AutoRegime.git
