# Demo 4: Custom Analysis with Seed
from autoregime import AutoRegimeDetector, MarketDataLoader
import numpy as np

np.random.seed(42)  # Consistency guarantee
loader = MarketDataLoader()
detector = AutoRegimeDetector()
data = loader.load_market_data(['AAPL'], start_date='2020-01-01')
detector.fit(data)
regime = detector.predict_current_regime(data)
print(f"AAPL: {regime}")