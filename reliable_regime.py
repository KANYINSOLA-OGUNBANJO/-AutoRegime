from autoregime import AutoRegimeDetector, MarketDataLoader
import numpy as np

def reliable_regime_analysis(symbol, seed=42):
    np.random.seed(seed)
    
    loader = MarketDataLoader()
    detector = AutoRegimeDetector()
    
    data = loader.load_market_data([symbol], start_date='2020-01-01')
    detector.fit(data)
    
    regime = detector.predict_current_regime(data)
    return regime

# Test it
print("Testing reliability...")
result1 = reliable_regime_analysis('SPY')
result2 = reliable_regime_analysis('SPY')
print("Result 1:", result1)
print("Result 2:", result2)
print("Same?", result1 == result2)