# test_reliability.py
from reliable_regime import reliable_regime_analysis

# Test 1: Same symbol, multiple runs
print("=== RELIABILITY TEST ===")
for i in range(3):
    result = reliable_regime_analysis('SPY')
    print(f"Run {i+1}: {result}")

# Test 2: Different seeds
print("\n=== DIFFERENT SEEDS ===")
for seed in [42, 100, 200]:
    result = reliable_regime_analysis('SPY', seed=seed)
    print(f"Seed {seed}: {result}")

# Test 3: Updated AutoRegime
from autoregime import reliable_quick_analysis
print("\n=== UPDATED AUTOREGIME ===")
result = reliable_quick_analysis('SPY')
print(f"Reliable result: {result}")