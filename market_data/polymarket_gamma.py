from __future__ import annotations
import os, json, math, logging, time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
try:
    # requests is tiny and commonly present; we add it to requirements.txt if missing
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # Allow tests to inject a fetcher

YES_LABELS = {"yes", "y", "up", "true", "1"}
NO_LABELS  = {"no", "n", "down", "false", "0"}

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _parse_dt_utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # common ISO with 'Z'
        s2 = s.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s2).astimezone(timezone.utc)
        except Exception:
            return None
    return None

def compute_days_to_expiry(end_iso_or_ts: Any, now: Optional[datetime]=None) -> Optional[float]:
    """
    Compute (expiry - now) in days, clamp to >= 0.
    Returns None if expiry cannot be parsed.
    """
    end_dt = _parse_dt_utc(end_iso_or_ts)
    if end_dt is None:
        return None
    now_dt = now or _now_utc()
    delta = (end_dt - now_dt).total_seconds() / 86400.0
    return max(0.0, delta)

def _parse_listish(v: Any) -> Optional[List[Any]]:
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        # try JSON-encoded list first
        try:
            x = json.loads(s)
            if isinstance(x, list):
                return x
        except Exception:
            pass
        # fallback: comma-separated
        return [t.strip() for t in s.split(",") if t.strip() != ""]
    return None

def _normalize_clob_token_ids(v: Any) -> List[str]:
    xs = _parse_listish(v)
    if not xs:
        return []
    out: List[str] = []
    for t in xs:
        out.append(str(t))
    return out

def _map_yes_no_indices(outcomes: List[str]) -> Optional[Tuple[int, int]]:
    if not outcomes or len(outcomes) < 2:
        return None
    # Only binary markets are supported here.
    if len(outcomes) != 2:
        return None
    o0 = str(outcomes[0]).strip().lower()
    o1 = str(outcomes[1]).strip().lower()
    # prefer explicit yes/no
    if o0 in YES_LABELS and o1 in NO_LABELS:
        return (0, 1)
    if o0 in NO_LABELS and o1 in YES_LABELS:
        return (1, 0)
    # allow UP/DOWN mapping => YES/NO
    if o0 in {"up"} and o1 in {"down"}:
        return (0, 1)
    if o0 in {"down"} and o1 in {"up"}:
        return (1, 0)
    return None

def _health_log(enabled: bool, msg: str, payload: Dict[str, Any]) -> None:
    if not enabled:
        return
    line = {"ts": int(time.time()), "msg": msg, "payload": payload}
    path = os.getenv("PM_HEALTH_PATH", "analysis/pm_gamma_sample.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")

def normalize_gamma_market(gm: Dict[str, Any], *, now: Optional[datetime]=None) -> Optional[Dict[str, Any]]:
    """
    Returns normalized dict with yes_price/no_price strictly from gamma outcomePrices.
    If parsing/mapping fails → return None (skip).
    """
    health = os.getenv("PM_HEALTH_DEBUG", "0") == "1"
    outcomes_raw = gm.get("outcomes")
    prices_raw   = gm.get("outcomePrices")

    outcomes = _parse_listish(outcomes_raw)
    prices   = _parse_listish(prices_raw)
    if not outcomes or not prices or len(outcomes) != len(prices):
        _health_log(health, "MISMATCHED_OUTCOME_LEN", {"id": gm.get("id"), "outcomes": outcomes_raw, "outcomePrices": prices_raw})
        return None

    # coerce prices to float and validate 0..1
    try:
        prices_f = [float(x) for x in prices]
    except Exception:
        _health_log(health, "PRICE_PARSE_ERROR", {"id": gm.get("id"), "outcomePrices": prices_raw})
        return None
    if any((p < 0.0 or p > 1.0 or math.isnan(p)) for p in prices_f):
        _health_log(health, "PRICE_RANGE_ERROR", {"id": gm.get("id"), "outcomePrices": prices_raw})
        return None

    idxs = _map_yes_no_indices([str(x) for x in outcomes])
    if idxs is None:
        _health_log(health, "MISMATCHED_OUTCOME_LABELS", {"id": gm.get("id"), "outcomes": outcomes_raw})
        return None
    i_yes, i_no = idxs
    yes = prices_f[i_yes]
    no  = prices_f[i_no]

    # 0..1 asserts (hard failure)
    if not (0.0 <= yes <= 1.0 and 0.0 <= no <= 1.0):
        _health_log(health, "PRICE_RANGE_ERROR_YN", {"id": gm.get("id"), "y": yes, "n": no})
        return None

    endDate = gm.get("endDate") or gm.get("endDateIso")
    dte = compute_days_to_expiry(endDate, now=now)

    out: Dict[str, Any] = {
        "pm_market_id": gm.get("id"),
        "pm_question": gm.get("question"),
        "conditionId": gm.get("conditionId"),
        "clobTokenIds": _normalize_clob_token_ids(gm.get("clobTokenIds")),
        "endDate": endDate,
        "days_to_expiry": dte,
        "yes_price": yes,
        "no_price": no,
        "price_source": "gamma_outcome",
        # Telemetry only
        "pm_last_trade_price": gm.get("lastTradePrice"),
        "pm_best_bid": gm.get("bestBid"),
        "pm_best_ask": gm.get("bestAsk"),
        # Raw breadcrumbs (helpful in audits)
        "_gamma_outcomes_raw": outcomes_raw,
        "_gamma_outcomePrices_raw": prices_raw,
    }
    _health_log(health, "NORMALIZED", {"id": out["pm_market_id"], "y": yes, "n": no})
    return out

def fetch_markets_from_gamma(url: str = "https://gamma-api.polymarket.com/markets",
                             fetcher=None,
                             query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Fetch raw markets from Gamma (no auth required).
    fetcher: Optional[Callable[[str], requests.Response]] for test injection.
    """
    if fetcher is None:
        if requests is None:
            raise RuntimeError("requests not available; install requirements first.")
        # Default to only live (open) markets at the HTTP layer
        params = {"closed": "false"}
        if query:
            params.update(query)
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    else:
        # Test hook: fetcher can ignore 'query' or consume it
        resp = fetcher(url) if query is None else fetcher(url, query)
        return resp.json() if hasattr(resp, "json") else resp

def _is_live_market(gm: Dict[str, Any]) -> bool:
    # Live = not closed, active true, not archived
    if gm.get("closed") is True:
        return False
    if gm.get("archived") is True:
        return False
    return gm.get("active") is True

def load_polymarket_gamma_normalized(now: Optional[datetime]=None,
                                     url: str="https://gamma-api.polymarket.com/markets",
                                     fetcher=None,
                                     live_only: bool = True) -> List[Dict[str, Any]]:
    # Query only open markets at the API level; we also post-filter for active/archived.
    raw = fetch_markets_from_gamma(url=url, fetcher=fetcher, query={"closed": "false"} if live_only else None)
    out: List[Dict[str, Any]] = []
    for gm in raw:
        if live_only and not _is_live_market(gm):
            _health_log(os.getenv("PM_HEALTH_DEBUG", "0") == "1",
                        "SKIP_NOT_LIVE",
                        {"id": gm.get("id"), "active": gm.get("active"), "closed": gm.get("closed"), "archived": gm.get("archived")})
            continue
        m = normalize_gamma_market(gm, now=now)
        if m is not None:
            out.append(m)
    return out