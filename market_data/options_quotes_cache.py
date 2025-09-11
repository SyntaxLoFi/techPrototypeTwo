from __future__ import annotations
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Any, Optional
import time, os, json, pathlib

@dataclass
class QuotesSnapshot:
    currency: str
    items: List[Dict[str, Any]]
    asof_ms: int = field(default_factory=lambda: int(time.time() * 1000))

class OptionsQuotesCache:
    """Process-local cache for enriched Lyra options quotes with optional snapshots."""
    _instance: "OptionsQuotesCache" = None
    _lock = RLock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data: Dict[str, QuotesSnapshot] = {}
            cls._instance._save = os.getenv("SAVE_SNAPSHOTS", "0").lower() in ("1","true","yes")
            cls._instance._dir = os.getenv("SNAPSHOT_DIR", "./data/snapshots")
        return cls._instance

    def put_many(self, currency: str, quotes: List[Dict[str, Any]]) -> QuotesSnapshot:
        snap = QuotesSnapshot(currency=currency, items=list(quotes))
        with self._lock:
            self._data[currency] = snap
        if self._save:
            self._persist_snapshot(snap)
        return snap

    def get_snapshot(self, currency: str) -> List[Dict[str, Any]]:
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

    def _persist_snapshot(self, snap: QuotesSnapshot) -> None:
        ts_dir = os.path.join(self._dir, str(snap.asof_ms))
        pathlib.Path(ts_dir).mkdir(parents=True, exist_ok=True)
        fname = os.path.join(ts_dir, f"{snap.currency}_lyra_options.json")
        norm: List[Dict[str, Any]] = []
        for q in snap.items:
            norm.append({
                "instrument_name": q.get("instrument_name") or q.get("symbol"),
                "currency": q.get("currency") or snap.currency,
                "expiry_ts": int(q.get("expiry_ts") or q.get("expiry") or 0),
                "strike": float(q.get("strike") or q.get("strike_price") or 0.0),
                "right": (q.get("type") or q.get("option_type") or "").upper()[0:1],
                "bid": _to_float(q.get("bid")),
                "ask": _to_float(q.get("ask")),
                "mid": _to_float(q.get("mid")),
                "iv_bid": _to_float(q.get("iv_bid")),
                "iv_ask": _to_float(q.get("iv_ask")),
                "greeks": q.get("greeks") or {},
                "source": q.get("source") or ("EST" if q.get("price_estimated") else "WS"),
                "ts": int(q.get("ts") or snap.asof_ms),
            })
        with open(fname, "w") as f:
            json.dump(norm, f, separators=(",", ":"))

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None