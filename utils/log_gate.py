import logging
import os
import threading
from typing import Optional, Any

# Module-level switches configured once per run
_enabled_reason = True
_reason_levelno = logging.DEBUG
_max_reason_lines = 400
_per_ccy_snapshot = True
_ev_summary_info = True

_count = 0
_truncated = False
_lock = threading.Lock()
_run_id = None

def _reset_if_new_run():
    global _count, _truncated, _run_id
    rid = os.getenv("APP_RUN_ID")
    if rid != _run_id:
        with _lock:
            _count = 0
            _truncated = False
            _run_id = rid

def configure_from_config(config: Optional[Any], run_id: Optional[str] = None) -> None:
    """Call once near startup."""
    global _enabled_reason, _reason_levelno, _max_reason_lines, _per_ccy_snapshot, _ev_summary_info, _run_id
    # env wins over YAML for level/flags
    dbg = getattr(config, "debug", None) if config else None
    log = getattr(dbg, "log", None) if dbg else None
    _enabled_reason = _get_bool_env("DEBUG_REASON_CODES", getattr(log, "reason_codes", True))
    level = os.getenv("DEBUG_REASON_LEVEL", getattr(log, "reason_codes_level", "DEBUG"))
    _reason_levelno = logging.DEBUG if str(level).upper() == "DEBUG" else logging.INFO
    try:
        _max_reason_lines = int(os.getenv("DEBUG_REASON_MAX_LINES", getattr(log, "reason_codes_max_lines", 400)))
    except Exception:
        _max_reason_lines = 400
    _per_ccy_snapshot = _get_bool_env("DEBUG_PER_CCY_SNAPSHOT", getattr(log, "per_currency_snapshot", True))
    _ev_summary_info = _get_bool_env("DEBUG_EV_SUMMARY_INFO", getattr(log, "ev_summary_info", True))
    if run_id:
        _run_id = run_id

def _get_bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return v.strip().lower() in {"1","true","t","yes","y","on"}

def reason_debug(logger: logging.Logger, msg: str, *args, **kwargs) -> None:
    """Gate reason-code logs: cap lines and support INFO/DEBUG level."""
    _reset_if_new_run()
    if not _enabled_reason:
        return
    global _count, _truncated
    with _lock:
        if _count < _max_reason_lines:
            logger.log(_reason_levelno, msg, *args, **kwargs)
            _count += 1
        elif not _truncated:
            logger.log(
                _reason_levelno,
                "REASONS_TRUNCATED after %d lines (raise debug.log.reason_codes_max_lines or set DEBUG_REASON_MAX_LINES).",
                _max_reason_lines,
            )
            _truncated = True

def per_currency_snapshot_enabled() -> bool:
    _reset_if_new_run()
    return _per_ccy_snapshot

def ev_summary_info_enabled() -> bool:
    _reset_if_new_run()
    return _ev_summary_info