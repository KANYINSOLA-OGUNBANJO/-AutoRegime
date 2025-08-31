# 🚀 AutoRegime: AI-Powered Market Regime Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoRegime** is an advanced AI system that automatically detects and analyzes market regimes using Hidden Markov Models. Built with 25+ years of market data validation, it identifies bull markets, bear markets, sideways trends, and crisis periods with professional-grade accuracy.

## ✨ Key Features

- 🤖 **AI-Powered Detection**: Hidden Markov Models with 6 sophisticated market indicators
- 📊 **25+ Years of Data**: Validated performance from 1995 to present
- 🎯 **Regime Classification**: Bull, Bear, Sideways, and Crisis detection
- 📈 **Professional Visualizations**: Interactive charts and regime timelines
- 🌐 **Web Dashboard**: Streamlit-powered interface for real-time analysis
- ⚡ **One-Line Usage**: Simple API for quick analysis

## 🚀 Quick Start

### Installation
```bash
pip install git+https://github.com/KANYINSOLA-OGUNBANJO/AutoRegime.git
```

### Basic Usage
```python
import autoregime

# Quick demo with SPY data
autoregime.quick_demo()

# Launch interactive dashboard
autoregime.launch_dashboard()

# Advanced usage
detector = autoregime.AutoRegimeDetector()
regimes = detector.detect_regimes('SPY', start_date='2020-01-01')
detector.plot_regimes()
```

## 📊 What AutoRegime Detects

- **Bull Markets**: Extended upward trends with high momentum
- **Bear Markets**: Sustained downward movements and high volatility  
- **Sideways Markets**: Range-bound trading with low directional bias
- **Crisis Periods**: Extreme volatility and market stress events

## 🎯 Real-World Validation

AutoRegime successfully identified major market events:
- ✅ COVID-19 Market Crash (March 2020)
- ✅ 2008 Financial Crisis
- ✅ Dot-com Bubble (2000-2002)
- ✅ Bull market runs (2009-2020, 2020-2021)

## 📈 Example Output

```python
# Detected Regimes for SPY (2020-2024)
Regime 1: Bull Market (2020-01 to 2020-02) - 68 days
Regime 2: Crisis (2020-03 to 2020-04) - 43 days  
Regime 3: Bull Market (2020-05 to 2022-01) - 612 days
Regime 4: Bear Market (2022-01 to 2022-10) - 284 days
Regime 5: Bull Market (2022-11 to present) - 456 days
```

## 🛠 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, plotly, streamlit
- yfinance for market data

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## 📧 Contact

Built by Kanyinsola Ogunbanjo - kanyinsolaogunbanjo@gmail.com