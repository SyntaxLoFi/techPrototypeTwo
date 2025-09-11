from __future__ import annotations

import os
import json
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Dict

try:
    from logger_config import set_request_id  # type: ignore
except Exception:
    def set_request_id(*args, **kwargs):  # type: ignore
        pass

# Dedicated logger for low-noise health lines
_log = logging.getLogger("DebugRecorder")

class RawDataRecorder:
    def __init__(self, *, enabled: bool = False, dump_dir: str = "debug_runs",
                 capture: Optional[Dict[str, bool]] = None, run_id: Optional[str] = None) -> None:
        self.enabled = bool(enabled)
        self.dump_dir = dump_dir
        self.capture = {"polymarket": True, "options": True, "perps": True, "snapshot": True, "checkpoint": True}
        if capture:
            self.capture.update(capture)
        self._written = set()
        self._lock = threading.Lock()

        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        if self.enabled:
            try:
                set_request_id(self.run_id)
            except Exception:
                pass
            # One-line health log so we can verify it's active
            try:
                _log.info(
                    "DebugRecorder enabled: run_id=%s dump_dir=%s capture=[%s]",
                    self.run_id,
                    str((Path(self.dump_dir) / self.run_id).resolve()),
                    ",".join(k for k, v in (self.capture or {}).items() if v),
                )
            except Exception:
                # Never fail the app because of logging
                pass

    def _base_dir(self) -> Path:
        p = Path(self.dump_dir) / self.run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def dump_json(self, rel_path: str, data: Any, *, category: str = "snapshot", overwrite: bool = False) -> Optional[str]:
        """
        Write a JSON file under the run directory.

        By default this recorder is write-once per (category, path). If `overwrite=True`,
        the write will proceed even if the file was already written in this run.
        Writes are atomic (tmp + os.replace).
        """
        if not self.enabled or not self.capture.get(category, True):
            return None
        key = f"{category}:{rel_path}"
        with self._lock:
            if key in self._written and not overwrite:
                return None
            out_path = self._base_dir() / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(out_path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=_json_default)
            os.replace(tmp, out_path)
            # record as written even if this was an overwrite, to avoid double writes later
            self._written.add(key)
            # Health line per write (once per file per run)
            try:
                if self.enabled:
                    _log.info("Saved debug dump [%s]: %s", category or "unknown", str(out_path))
            except Exception:
                pass
            return str(out_path)

_recorder: Optional[RawDataRecorder] = None
_rec_lock = threading.Lock()

def get_recorder(config: Optional[Any] = None) -> RawDataRecorder:
    global _recorder
    with _rec_lock:
        if _recorder is not None:
            return _recorder
        # Defaults; allow config/env to override
        enabled = False
        dump_dir = os.getenv("DEBUG_DUMP_DIR", "debug_runs")
        capture: Dict[str, bool] = {}
        # Pull from config if provided
        try:
            dbg = getattr(config, "debug", None) if config else None
            if dbg:
                enabled = bool(getattr(dbg, "enabled", enabled))
                dump_dir = getattr(dbg, "dump_dir", dump_dir)
                cap = getattr(dbg, "capture", None)
                if isinstance(cap, dict):
                    capture.update(cap)
        except Exception:
            pass
        # Env overrides
        enabled = _env_bool("DEBUG", enabled)
        for cat in ("polymarket","options","perps","snapshot","checkpoint"):
            env = os.getenv(f"DEBUG_CAPTURE_{cat.upper()}")
            if env:
                capture[cat] = env.strip().lower() in {"1","true","t","yes","y","on"}

        run_id = os.getenv("APP_RUN_ID")
        _recorder = RawDataRecorder(enabled=enabled, dump_dir=dump_dir, capture=capture, run_id=run_id)
        return _recorder

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in {"1","true","t","yes","y","on"}

def _json_default(obj: Any) -> Any:
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
    except Exception:
        pass
    try:
        from datetime import datetime, date
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
    except Exception:
        pass
    try:
        return str(obj)
    except Exception:
        return None