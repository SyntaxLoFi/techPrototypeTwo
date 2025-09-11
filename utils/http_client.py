# utils/http_client.py
from __future__ import annotations

import os
import time
import threading
from collections import deque
from typing import Iterable, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TPS = int(os.getenv("REST_TPS_LIMIT", "10"))          # Derive: 10 TPS (public REST)
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SEC", "10"))
HTTP_RETRY_MAX = int(os.getenv("HTTP_RETRY_MAX", "3"))
HTTP_BACKOFF = float(os.getenv("HTTP_BACKOFF", "0.2"))

class RateLimiter:
    """Simple fixed-window-ish limiter suitable for one-process clients."""
    def __init__(self, max_calls: int, period: float = 1.0):
        self.max_calls = max_calls
        self.period = period
        self._calls = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            # Drop old timestamps
            while self._calls and (now - self._calls[0]) >= self.period:
                self._calls.popleft()
            # Sleep if full
            sleep_for = 0.0
            if len(self._calls) >= self.max_calls:
                sleep_for = self.period - (now - self._calls[0])
        if sleep_for > 0:
            time.sleep(sleep_for)
        with self._lock:
            self._calls.append(time.monotonic())

def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=HTTP_RETRY_MAX,
        backoff_factor=HTTP_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"Content-Type": "application/json"})
    return s

session = _build_session()
limiter = RateLimiter(DEFAULT_TPS, period=1.0)

def get(url: str, **kwargs):
    limiter.wait()
    kwargs.setdefault("timeout", HTTP_TIMEOUT)
    return session.get(url, **kwargs)

def post(url: str, **kwargs):
    limiter.wait()
    kwargs.setdefault("timeout", HTTP_TIMEOUT)
    return session.post(url, **kwargs)