"""Strategy Tag Router
Maps hierarchical strategy tags (e.g., "options.variance") to strategy classes.

Resilient to StrategyLoader API differences:
 - If the loader exposes `.discover()`, we'll call it.
 - Otherwise we rely on the loader's eager `_load_all_strategies()` done in __init__.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Type
import inspect
import logging

try:
    from .base_strategy import BaseStrategy  # type: ignore
except (ImportError, ValueError):
    from base_strategy import BaseStrategy  # type: ignore

try:
    from .strategy_loader import StrategyLoader  # type: ignore
except (ImportError, ValueError):
    from strategy_loader import StrategyLoader  # type: ignore

_logger = logging.getLogger("StrategyTagRouter")

class StrategyTagRouter:
    def __init__(self, loader: Optional[StrategyLoader] = None) -> None:
        self.loader = loader or StrategyLoader()
        self._tag_to_class: Dict[str, Type[BaseStrategy]] = {}
        self._discovered: bool = False

    def _ensure_loaded(self) -> None:
        """Best-effort discovery: call discover() if present; otherwise rely on loader.__init__."""
        if self._discovered:
            return
        try:
            discover = getattr(self.loader, "discover", None)
            if callable(discover):
                discover()
        except Exception:
            pass
        self._discovered = True

    def _build_index(self) -> None:
        self._ensure_loaded()
        # Prefer loader.strategies (eagerly populated in this repo)
        strategies: Dict[str, Type[BaseStrategy]] = {}
        try:
            strategies = getattr(self.loader, "strategies", {}) or {}
        except Exception:
            strategies = {}
        if not strategies:
            try:
                strategies = dict(self.loader.get_all_strategies())
            except Exception:
                strategies = {}
        for _, cls in strategies.items():
            try:
                tags = getattr(cls, "TAGS", None)
                if not tags:
                    continue
                for tag in tags:
                    if isinstance(tag, str):
                        self._tag_to_class.setdefault(tag, cls)
            except Exception as e:
                _logger.debug("Skipping during tag index: %s", e)

    # Compatibility shim if other code calls router.discover()
    def discover(self) -> None:
        if not self._tag_to_class:
            self._build_index()

    def get_supported_tags(self) -> List[str]:
        if not self._tag_to_class:
            self._build_index()
        return sorted(self._tag_to_class.keys())

    def instantiate_for_tags(self, tags: Sequence[str], *, risk_free_rate: Optional[float] = None, cfg: Optional[object] = None) -> List[BaseStrategy]:
        if not self._tag_to_class:
            self._build_index()
        instances: List[BaseStrategy] = []
        seen: set = set()
        for tag in tags or []:
            cls = self._tag_to_class.get(tag)
            if not cls or cls in seen:
                continue
            seen.add(cls)
            try:
                sig = inspect.signature(cls.__init__)
                params = sig.parameters
                kwargs = {}
                if 'risk_free_rate' in params and risk_free_rate is not None:
                    kwargs['risk_free_rate'] = risk_free_rate
                if 'cfg' in params and cfg is not None:
                    kwargs['cfg'] = cfg
                if 'config' in params and cfg is not None:
                    kwargs['config'] = cfg
                if 'logger' in params:
                    kwargs['logger'] = logging.getLogger(f"strategy.{cls.__name__}")
                instances.append(cls(**kwargs) if kwargs else cls())
            except Exception as e:
                _logger.warning("Failed to instantiate %s for tag %s: %s", cls, tag, e)
        return instances