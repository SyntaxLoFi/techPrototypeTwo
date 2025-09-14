import json, os, threading
from datetime import datetime, timezone
_lock = threading.Lock()

def _enabled() -> bool:
    return os.getenv("VALIDATION_AUDIT_ENABLE", "0") == "1"

def _path() -> str:
    return os.getenv("VALIDATION_AUDIT_PATH", "analysis/validation_audit.jsonl")

def emit(obj: dict) -> None:
    if not _enabled():
        return
    path = _path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = dict(obj or {})
    obj.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with _lock:
        with open(path, "a") as fh:
            fh.write(json.dumps(obj, separators=(",", ":")) + "\n")