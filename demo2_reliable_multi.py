# Reliable Multi-Asset Analysis
from reliable_regime import reliable_multi_asset
results = reliable_multi_asset(['SPY', 'QQQ', 'TLT'])
for symbol, result in results.items():
    print(f"{symbol}: {result}")