"""
Strategies Package - Modular arbitrage strategies
"""

try:
    from .base_strategy import BaseStrategy
except (ImportError, ValueError):
    from base_strategy import BaseStrategy

__all__ = ['BaseStrategy']