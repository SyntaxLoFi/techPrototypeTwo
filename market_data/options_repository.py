from __future__ import annotations
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Any, Optional
import json, os, time

@dataclass
class OptionSnapshot:
    currency: str
    items: List[Dict[str, Any]]
    asof: float = field(default_factory=lambda: time.time())

class OptionsRepository:
    """
    Minimal, process-local repository for Lyra option snapshots.
    Persist-to-disk for debugging with:
      - export LYRA_SAVE_SNAPSHOTS=1
      - export LYRA_SNAPSHOT_DIR=/path/to/dir
    """
    _instance: "OptionsRepository" = None
    _lock = RLock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data: Dict[str, OptionSnapshot] = {}
            cls._instance._snapshot_dir = os.environ.get("LYRA_SNAPSHOT_DIR")
            cls._instance._save_snapshots = os.environ.get("LYRA_SAVE_SNAPSHOTS", "0").lower() in ("1","true","yes")
        return cls._instance

    def set_active(self, currency: str, items: List[Dict[str, Any]]) -> OptionSnapshot:
        snap = OptionSnapshot(currency=currency, items=list(items))
        with self._lock:
            self._data[currency] = snap
        if self._save_snapshots and self._snapshot_dir:
            os.makedirs(self._snapshot_dir, exist_ok=True)
            path = os.path.join(self._snapshot_dir, f"lyra_{currency}_{int(snap.asof)}.json")
            with open(path, "w") as f:
                json.dump(items, f, separators=(",",":"))
        return snap

    def get_active(self, currency: str) -> List[Dict[str, Any]]:
        with self._lock:
            snap = self._data.get(currency)
            return list(snap.items) if snap else []

    def has_currency(self, currency: str) -> bool:
        with self._lock:
            snap = self._data.get(currency)
            return bool(snap and snap.items)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()