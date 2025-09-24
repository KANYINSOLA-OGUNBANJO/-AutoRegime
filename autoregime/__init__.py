# autoregime/__init__.py
"""
AutoRegime public API.
This exposes the stable_regime_analysis function from the improved sticky-HMM engine.
"""

from .engines.hmm_sticky import stable_regime_analysis, stable_report

__all__ = ["stable_regime_analysis"]