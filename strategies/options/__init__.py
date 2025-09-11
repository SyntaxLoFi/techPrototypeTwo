"""
Options Strategies Package - All options-based hedging strategies.

This package intentionally avoids importing individual strategy modules at
package import time to keep dynamic discovery working even when some strategy
files are absent. Use strategies.strategy_loader to discover strategies.
"""
try:
    from .base_options_strategy import BaseOptionsStrategy
except Exception:
    # For direct execution contexts
    from base_options_strategy import BaseOptionsStrategy  # type: ignore

__all__ = ["BaseOptionsStrategy"]
