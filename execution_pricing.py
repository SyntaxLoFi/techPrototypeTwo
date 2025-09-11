"""Shared execution-pricing helpers for options and prediction markets.

This module centralizes *execution plumbing* (fees, slippage) so strategies
can keep their academic math intact while speaking a single cost language.

- Options legs use OPTION_SLIPPAGE_BPS and OPTION_FEE_BPS
- Prediction-market legs use PM_SLIPPAGE_BPS and PM_FEE_BPS

All helpers preserve sign conventions:
- Debits are positive cash outflows
- Credits are negative (returned) cashflows
"""
from typing import Literal, Dict, Any, Sequence
from math import isfinite
from datetime import datetime, timezone
from config_manager import (
    OPTION_SLIPPAGE_BPS,
    OPTION_FEE_BPS,
    PM_SLIPPAGE_BPS,
    PM_FEE_BPS,
    OPTIONS_UNWIND_MODEL,
    RISK_FREE_RATE,
)
from black_scholes_greeks import BlackScholesGreeks as _BS

Side = Literal['BUY','SELL','buy','sell']

def option_apply_fee(*_args, **_kwargs):
    """Deprecated: do not call. Fees already included via option_exec_price().
    Kept only as a guard to surface accidental double-charging at runtime."""
    raise RuntimeError("Deprecated: fees are included in option_exec_price(); do not call option_apply_fee().")


def _bp(bps: float) -> float:
    return float(bps) / 10_000.0

def pm_execute_price(side: Side, price_0_1: float) -> float:
    """Executable PM ticket price in [0,1] with slippage+fees applied.

    This is an execution helper (not a valuation). It does **not** change the
    semantics of the strategy; it standardizes how we translate quoted PM prices
    into executable cashflows.
    """
    side = str(side).lower()
    base = float(price_0_1)
    if not (0.0 <= base <= 1.0):
        base = max(0.0, min(1.0, base))
    slip = _bp(PM_SLIPPAGE_BPS)
    fee  = _bp(PM_FEE_BPS)
    if side == 'buy':
        return base * (1.0 + slip + fee)
    elif side == 'sell':
        return base * (1.0 - slip - fee)
    else:
        raise ValueError("side must be 'buy' or 'sell'")

def pm_exit_price(side: Side, settle_0_1: float) -> float:
    """Mirror entry fees at closeout; modeled as a sell or buy at settlement."""
    # reuse pm_execute_price; semantics are symmetric
    return pm_execute_price(side, settle_0_1)

# --- Option execution helpers (entry & exit use same bps) ---
def option_exec_price(side: Side, bid: float | None, ask: float | None, *, mid: float | None = None) -> float:
    """
    Convert quotes to an executable price with slippage+fees.
    - Debits are positive (buy), credits negative (sell).
    """
    side = str(side).lower()
    b = float(bid) if isfinite(bid or float("nan")) else None
    a = float(ask) if isfinite(ask or float("nan")) else None
    m = float(mid) if isfinite(mid or float("nan")) else None
    fee = _bp(OPTION_FEE_BPS)
    slip = _bp(OPTION_SLIPPAGE_BPS)
    if side == "buy":
        if a is not None and a > 0:
            return a * (1.0 + fee + slip)
        if m is not None and m > 0:
            return m * (1.0 + fee + slip)
        if b is not None and b > 0:
            return b * (1.0 + fee + slip)  # conservative
    elif side == "sell":
        if b is not None and b > 0:
            return -b * (1.0 - fee - slip)
        if m is not None and m > 0:
            return -m * (1.0 - fee - slip)
        if a is not None and a > 0:
            return -a * (1.0 - fee - slip)  # conservative
    raise ValueError("Option quote(s) missing or invalid")

def _bs_price(otype: str, S: float, K: float, r: float, sigma: float, T: float) -> float:
    if otype.upper() == "CALL":
        return _BS.call_price(S, K, r, sigma, T)
    return _BS.put_price(S, K, r, sigma, T)

def option_closeout_value_at_pm(
    legs: Sequence[Dict[str, Any]],
    is_above: bool,
    is_yes_state: bool,
    *,
    spot_at_pm: float | None = None,
    r: float | None = None,
) -> float:
    """
    Value the option legs at PM resolution (T_pm), including *close-out* fees.
    Configured by OPTIONS_UNWIND_MODEL:
      - "sticky_strike": BS with original strike IV, reduced time; optional IV drop
      - "intrinsic_only": intrinsic at PM time
    Close-out = opposite of entry action.
    Returns signed USD (credits negative, debits positive).
    """
    if not legs:
        return 0.0
    model = str(OPTIONS_UNWIND_MODEL).lower()
    r = RISK_FREE_RATE if r is None else float(r)
    total = 0.0
    for leg in legs:
        otype  = (leg.get("type") or "CALL").upper()
        K      = float(leg.get("strike", 0.0) or 0.0)
        expiry = leg.get("expiry")
        iv     = leg.get("iv")
        qty    = float(leg.get("contracts", 0.0) or 0.0) * float(leg.get("contract_size", 1.0) or 1.0)
        entry_action = (leg.get("action") or "BUY").upper()

        # Opposite side to flatten at T_pm
        close_side = "SELL" if entry_action == "BUY" else "BUY"

        if model == "intrinsic_only" or iv is None or expiry is None:
            # Binary state at PM time: ITM legs intrinsic, OTM zero
            # Choose a state spot that triggers the PM truth.
            # If ABOVE is true state -> S slightly >= K; else slightly below.
            if is_yes_state:
                S = max(K + 1e-8, float(spot_at_pm or K))
            else:
                S = min(K - 1e-8, float(spot_at_pm or K))
            intrinsic = max(0.0, S - K) if otype == "CALL" else max(0.0, K - S)
            total += option_exec_price(close_side, None, None, mid=intrinsic) * qty
            continue

        # sticky_strike: same per-strike IV, reduced T
        try:
            from datetime import datetime
            # interpret expiry as YYYY-MM-DD
            dt_exp = datetime.fromisoformat(str(expiry))
        except Exception:
            dt_exp = None
        # T > 0 by floor; we value immediately at PM resolution with shrinkage
        T = max(1e-6, ((dt_exp - datetime.utcnow()).days) / 365.25) if dt_exp else 1e-6
        sigma = float(iv)
        # optional post-event vol drop handled by caller via leg['iv'], so keep here as-is
        # Use a state spot consistent with PM truth
        S = float(spot_at_pm or K)
        S = max(K + 1e-8, S) if is_yes_state else min(K - 1e-8, S)
        theo = _bs_price(otype, S, K, r, sigma, T)
        total += option_exec_price(close_side, None, None, mid=theo) * qty
    return total

def violates_no_immediate_loss(execution_details: Dict[str, Any]) -> bool:
    """
    Return True if the entry cashflow (PM ticket + option legs) is a *net debit*,
    i.e., the strategy requires paying upfront with executable prices after fees/slippage.

    Conventions:
      • Debits are positive; credits are negative (see option_exec_price and pm_execute_price).
      • Missing quotes or malformed legs => conservative True (violation).
    """
    try:
        ed = execution_details or {}
        total = 0.0

        # Polymarket ticket (YES by default)
        pm = ed.get("polymarket") or {}
        side = (pm.get("side") or "YES").upper()
        price = pm.get("price")
        qty = float(pm.get("qty") or 0.0)
        if price and qty:
            side_exec = "buy" if side in ("YES", "BUY") else "sell"
            total += pm_execute_price(side_exec, float(price)) * qty

        # Options legs
        for leg in ed.get("options_legs") or []:
            action = (leg.get("action") or leg.get("side") or "").lower()
            if action not in ("buy", "sell"):
                continue
            bid = leg.get("bid")
            ask = leg.get("ask")
            mid = leg.get("mark")
            qty = float(leg.get("contracts") or 0.0) * float(leg.get("contract_size") or 1.0)
            if qty == 0.0:
                continue
            price = option_exec_price(action, bid, ask, mid=mid)
            total += price * qty

        # Net debit (> 0) violates the "no-immediate-loss" entry constraint
        return (total > 1e-9)
    except Exception:
        # Be conservative if anything is missing/unparseable
        return True