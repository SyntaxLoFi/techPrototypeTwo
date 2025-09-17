# Expiry Selection: Bug Fix & Multi-Expiry Enumeration

**Problem.** The pipeline selected expiries by calendar-day buckets:
1) `filters/option_expiry.filter_options_by_expiry` used `now.date()` and compared `abs((expiry - today).days - pm_days)`, causing <1d PM horizons to prefer **tomorrow** over **today**.
2) `pm.PMContract.candidate_expiries` discarded time-of-day and ranked expiries by integer-day distance from the parsed PM point.
3) YAML policy keys (`hedging.variance.expiry_policy/max_expiry_gap_days/max_expiries_considered`) existed but were not wired to selection.

**Fixes (this PR).**
- **Hour-precise DTE** in `filter_options_by_expiry` (no ceil/floor).
- **Time-of-day respected** in `PMContract.candidate_expiries` (rank by hour distance).
- **New shared enumerator** `core/expiry_window.enumerate_expiries`:
  - Enumerates **all expiries** within [0h .. max_expiry_gap_days*24h] after PM settlement timestamp.
  - Applies liquidity gates (`min_quotes_per_expiry`, `min_strikes_required`).
  - Returns ordered candidates with metadata (weekly/monthly, dte_hours, why_kept).
- Variance-swap strategy now prefers the shared enumerator (graceful fallback keeps public API stable).

**Configuration.**
- Default policy mirrors schema defaults: `expiry_policy="allow_far_with_unwind"`, `max_expiry_gap_days=60`, `max_expiries_considered=10`.
- Date-only PM markets anchor to **23:59:59Z** by default in the enumerator.

**Observability.**
- Structured logs to `debug_runs/expiry_debug.jsonl`: PM ts, all candidates, keps, reasons.
- `filter_options_by_expiry` now records hour-level DTE samples and selection distances.

**Guardrails.**
- No silent "skip" of same-day PM unless a configured fallback authorizes it.
- All datetime comparisons are timezone-aware.
- Public strategy return shape unchanged.

**How to verify.**
1) Run the scanner for PM dates around Sep-16..19 2025 and inspect `debug_runs/expiry_debug.jsonl`.
2) Confirm same-day expiries appear when PM horizons are <24h and that candidates cover 0..60d.
3) Run `pytest -q`: see `tests/test_expiry_selection.py` for unit coverage.

**Rollback plan.**
- Revert `core/expiry_window.py`, `filters/option_expiry.py`, `pm/contract.py`, and `strategies/variance_swap.py` changes.
- The system will fall back to legacy single-expiry selection with integer-day buckets.