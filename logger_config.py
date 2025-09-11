# logger_config.py
import json
import logging
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Optional

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

def set_request_id(value: Optional[str]) -> None:
    _request_id.set(value or "-")

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", _request_id.get()),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "pid": os.getpid(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True

def _env_bool(name: str, default: bool) -> bool:
    return (os.getenv(name, str(default)) or str(default)).strip().lower() in {"1","true","t","yes","y","on"}

def setup_logging(*,
                  level: Optional[str] = None,
                  to_file: Optional[bool] = None,
                  json_logs: Optional[bool] = None,
                  filename: Optional[str] = None,
                  max_bytes: int = 20 * 1024 * 1024,
                  backup_count: int = 5,
                  utc: bool = True) -> None:
    """
    Idempotent logging setup. You can pass values from your validated config or rely on env vars:
      LOG_LEVEL, LOG_TO_FILE, LOG_JSON, LOG_FILENAME, LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOG_UTC
    """
    if getattr(setup_logging, "_configured", False):
        return

    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    to_file = to_file if to_file is not None else _env_bool("LOG_TO_FILE", True)
    # Default to human-friendly console unless explicitly overridden.
    json_logs = json_logs if json_logs is not None else _env_bool("LOG_JSON", False)
    file_json = _env_bool("LOG_FILE_JSON", True)
    filename = filename or os.getenv("LOG_FILENAME", "logs/app.log")
    try:
        max_bytes = int(os.getenv("LOG_MAX_BYTES", str(max_bytes)))
        backup_count = int(os.getenv("LOG_BACKUP_COUNT", str(backup_count)))
    except Exception:
        pass
    utc = utc if utc is not None else _env_bool("LOG_UTC", True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, lvl, logging.INFO))

    # Clear existing handlers for true idempotence
    for h in list(root.handlers):
        root.removeHandler(h)

    # Build both text and JSON formatters; choose per handler.
    text_fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s (req=%(request_id)s)",
        datefmt="%Y-%m-%dT%H:%M:%SZ" if utc else "%Y-%m-%dT%H:%M:%S%z",
    )
    if utc:
        class UtcFormatter(logging.Formatter):
            converter = staticmethod(lambda *args: datetime.now(timezone.utc).timetuple())
        text_fmt = UtcFormatter(text_fmt._fmt, text_fmt.datefmt)
    json_fmt = JsonFormatter()
    fmt_console = json_fmt if json_logs else text_fmt

    req_filter = RequestIdFilter()

    sh = logging.StreamHandler()
    sh.setFormatter(fmt_console)
    sh.addFilter(req_filter)
    root.addHandler(sh)

    if to_file:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fh = RotatingFileHandler(filename, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        fh.setFormatter(json_fmt if (json_logs or file_json) else text_fmt)
        fh.addFilter(req_filter)
        root.addHandler(fh)

    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    setup_logging._configured = True