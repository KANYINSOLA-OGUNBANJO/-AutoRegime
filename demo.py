#!/usr/bin/env python3
"""
AutoRegime Demo Script
Demonstrates basic usage of the AutoRegime system
"""

import autoregime

def main():
    print("🚀 AutoRegime Demo Starting...")
    print("=" * 50)
    
    # Quick demonstration
    print("Running quick demo with SPY data...")
    try:
        autoregime.quick_demo()
        print("✅ Demo completed successfully!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    print("\n" + "=" * 50)
    print("Demo finished. Try launching the dashboard with:")
    print("python -c 'import autoregime; autoregime.launch_dashboard()'")

if __name__ == "__main__":
    main()
