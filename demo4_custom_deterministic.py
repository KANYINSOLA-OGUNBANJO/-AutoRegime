# Custom Deterministic Analysis
from autoregime import AutoRegimeDetector, MarketDataLoader
import numpy as np

np.random.seed(42)  # Guaranteed consistency
detector = AutoRegimeDetector(random_state=42)
loader = MarketDataLoader()
data = loader.load_market_data(['AAPL'], start_date='2020-01-01')
detector.fit(data)
regime = detector.predict_current_regime(data)
print(f"AAPL: {regime}")