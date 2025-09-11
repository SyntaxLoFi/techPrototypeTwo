from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Iterable
import math
import re
import numpy as np

# -------------------- name helpers --------------------

def _is_option_name(nm: str) -> bool:
    n = (nm or "").lower()
    return (
        n.startswith("call_") or n.startswith("put_") or
        n.startswith("c_")    or n.startswith("p_")    or
        ":c:" in n            or ":p:" in n            or
        "_call_" in n         or "_put_" in n          or
        "call" in n           or "put" in n
    )

def _is_base_name(nm: str, base_names: Tuple[str, ...]) -> bool:
    return nm in base_names


# -------------------- core: OMP + exact refit --------------------

def select_sparse_quadratic(
    X: np.ndarray,
    y: np.ndarray,
    price_vec: np.ndarray,
    names: List[str],
    *,
    max_option_legs: int = 6,
    l2: float = 1e-8,
    budget: Optional[float] = 0.0,
    always_on_bases: Tuple[str, ...] = ("bond_T1", "S_T"),
    candidate_idx: Optional[Iterable[int]] = None,
    initial_active_idx: Optional[Iterable[int]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Greedy sparse selection (OMP) with exact KKT refit after each pick.
    Objective: (1/n)||Xw - y||^2 + l2||w||^2,  s.t. p^T w = budget (if not None)
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    p = np.asarray(price_vec, float).reshape(-1)

    n, k = X.shape
    if len(names) != k or len(p) != k:
        raise ValueError("X, names, price_vec must align")

    option_mask = np.array([_is_option_name(nm) for nm in names], dtype=bool)
    base_mask   = np.array([_is_base_name(nm, always_on_bases) for nm in names], dtype=bool)

    # Restrict the option *pool* to candidate_idx (if provided)
    if candidate_idx is not None:
        pool_mask = np.zeros(k, dtype=bool)
        for idx in candidate_idx:
            if 0 <= idx < k:
                pool_mask[idx] = True
        option_mask = option_mask & pool_mask

    # Normal equations precompute (ridge)
    G = (X.T @ X) / float(n)
    c = (X.T @ y) / float(n)
    G_reg = G + float(l2) * np.eye(k)

    def _refit_on(A: List[int]) -> np.ndarray:
        A_sorted = sorted(A)
        GA = G_reg[np.ix_(A_sorted, A_sorted)]
        cA = c[A_sorted]
        if budget is None:
            return np.linalg.solve(GA, cA)
        pA = p[A_sorted]
        KKT = np.zeros((len(A_sorted) + 1, len(A_sorted) + 1), float)
        KKT[:len(A_sorted), :len(A_sorted)] = GA
        KKT[:len(A_sorted), -1]             = pA
        KKT[-1, :len(A_sorted)]             = pA
        rhs = np.zeros(len(A_sorted) + 1, float)
        rhs[:len(A_sorted)] = cA
        rhs[-1]             = float(budget)
        sol = np.linalg.solve(KKT, rhs)
        return sol[:len(A_sorted)]

    # Initialize with bases + warm-start seeds
    A: List[int] = list(np.where(base_mask)[0])
    if initial_active_idx:
        for i in initial_active_idx:
            if 0 <= i < k and i not in A:
                A.append(i)

    # Initial fit
    w = np.zeros(k, float)
    if A:
        wA = _refit_on(A)
        for j, i in enumerate(sorted(A)):
            w[i] = wA[j]
    resid = y - X @ w

    # Option pool = eligible candidates not already active
    pool = [i for i in np.where(option_mask)[0] if i not in A]

    # Count *all* active options (seeded or not) toward the cap
    current_opt = sum(1 for i in A if _is_option_name(names[i]))
    remaining = max(0, int(max_option_legs) - current_opt)

    mse_path: List[float] = []
    for _ in range(remaining):
        if not pool:
            break
        # Pick by absolute correlation with residual
        corr = (X.T @ resid) / float(n)
        best_i = max(pool, key=lambda i: abs(corr[i]))
        A.append(best_i)

        # Exact refit on the active set
        w = np.zeros(k, float)
        wA = _refit_on(A)
        for j, i in enumerate(sorted(A)):
            w[i] = wA[j]

        resid = y - X @ w
        mse = float(np.mean(resid ** 2))
        mse_path.append(mse)
        if verbose:
            sel_count = sum(1 for i in A if _is_option_name(names[i]))
            print(f"[OMP] + {names[best_i]:<24s} | legs={sel_count} | MSE={mse:.6g}")

        pool.remove(best_i)

    mse_final = float(np.mean((X @ w - y) ** 2))
    info = {
        "active_idx": sorted(A),
        "mse_path": mse_path,
        "mse_final": mse_final,
        "selected_names": [names[i] for i in sorted(A)],
    }
    return w, info


# -------------------- rounding + repair --------------------

def round_options_and_repair_budget(
    names: List[str],
    w: np.ndarray,
    price_vec: np.ndarray,
    *,
    underlyings: Optional[List[str]] = None,
    asset_resolver: Optional[Dict[str, str]] = None,
    step_by_asset: Optional[Dict[str, float]] = None,
    floor_by_asset: Optional[Dict[str, float]] = None,
    default_step: float = 0.01,
    default_floor: float = 0.01,
    min_notional: float = 25.0,
    bond_name: str = "bond_T1",
) -> np.ndarray:
    """
    Quantize option legs to venue increments, drop tiny notionals, and
    repair budget by adjusting the bond.
    """
    wq = np.asarray(w, float).copy()
    p  = np.asarray(price_vec, float).reshape(-1)
    step_by_asset  = step_by_asset or {}
    floor_by_asset = floor_by_asset or {}

    def _asset_for(i: int) -> str:
        if underlyings is not None and i < len(underlyings) and underlyings[i]:
            return underlyings[i]
        if asset_resolver and names[i] in asset_resolver:
            return asset_resolver[names[i]]
        nm = (names[i] or "").upper()
        keys = set(step_by_asset.keys()) | set(floor_by_asset.keys())
        for key in sorted(keys, key=len, reverse=True):
            if key and key.upper() in nm:
                return key.upper()
        m = re.match(r"([A-Z]+)", nm)
        return m.group(1) if m else ""

    target_cost = float(p @ wq)

    for i, nm in enumerate(names):
        if _is_option_name(nm):
            asset = _asset_for(i)
            step  = float(step_by_asset.get(asset,  default_step))
            floor = float(floor_by_asset.get(asset, default_floor))

            if step > 0.0:
                wq[i] = step * np.round(wq[i] / step)
            if abs(wq[i]) < floor:
                wq[i] = 0.0
            if abs(wq[i] * p[i]) < float(min_notional):
                wq[i] = 0.0

    if bond_name in names:
        ib = names.index(bond_name)
        delta = target_cost - float(p @ wq)
        pbond = float(p[ib]) or 1.0
        wq[ib] += delta / pbond

    return wq


# -------------------- (optional) refit bases after rounding --------------------

def refit_bases_given_fixed_options(
    X: np.ndarray,
    y: np.ndarray,
    price_vec: np.ndarray,
    names: List[str],
    w_fixed: np.ndarray,
    *,
    l2: float = 1e-8,
    budget: Optional[float] = 0.0,
    base_names: Tuple[str, ...] = ("bond_T1", "S_T"),
) -> np.ndarray:
    """
    After rounding the option legs, improve the hedge by **refitting base columns only**
    (e.g., bond_T1 and S_T) while keeping option weights w_fixed frozen.

    Solves:
      minimize (1/n)||X_base w_base + X_opt w_fixed_opt - y||^2 + l2||w_base||^2
      s.t.     p_base^T w_base = budget - p_opt^T w_fixed_opt   (if budget is not None)
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    p = np.asarray(price_vec, float).reshape(-1)
    w_fixed = np.asarray(w_fixed, float)

    n, k = X.shape
    base_idx = [i for i, nm in enumerate(names) if _is_base_name(nm, base_names)]
    opt_idx  = [i for i, nm in enumerate(names) if _is_option_name(nm)]

    Xb = X[:, base_idx]
    Xo = X[:, opt_idx]

    y_tilde = y - Xo @ w_fixed[opt_idx]
    pb = p[base_idx]

    # Normal equations for bases
    Gb = (Xb.T @ Xb) / float(n) + float(l2) * np.eye(len(base_idx))
    cb = (Xb.T @ y_tilde) / float(n)

    if budget is None:
        wb = np.linalg.solve(Gb, cb)
    else:
        rhs_budget = float(budget) - float(p[opt_idx] @ w_fixed[opt_idx])
        KKT = np.zeros((len(base_idx)+1, len(base_idx)+1), float)
        KKT[:len(base_idx), :len(base_idx)] = Gb
        KKT[:len(base_idx), -1]             = pb
        KKT[-1, :len(base_idx)]             = pb
        rhs = np.zeros(len(base_idx)+1, float)
        rhs[:len(base_idx)] = cb
        rhs[-1]             = rhs_budget
        sol = np.linalg.solve(KKT, rhs)
        wb = sol[:len(base_idx)]

    w_new = w_fixed.copy()
    for j, i in enumerate(base_idx):
        w_new[i] = wb[j]
    return w_new


# -------------------- (optional) candidate strike preselection --------------------

def preselect_strikes_by_moneyness(
    strikes: List[float],
    S0: float,
    T_years: float,
    sigma: float,
    *,
    center_strike: Optional[float] = None,
    sigma_width: float = 2.0,
    max_candidates: Optional[int] = 24
) -> List[float]:
    """
    Keep strikes whose log-moneyness is within ±sigma_width * sigma * sqrt(T).
    Sort by proximity to center (ATM or provided center_strike) and truncate to max_candidates.
    """
    if T_years <= 0 or sigma <= 0 or S0 <= 0:
        return strikes
    center = float(center_strike) if center_strike else S0
    band = float(sigma_width) * float(sigma) * math.sqrt(float(T_years))
    m0 = math.log(max(center, 1e-12) / max(S0, 1e-12))
    scored = [(abs(math.log(k / max(S0, 1e-12)) - m0), k) for k in strikes if k > 0]
    scored.sort(key=lambda t: t[0])
    kept = [k for d, k in scored if d <= band]
    if max_candidates and len(kept) > max_candidates:
        kept = kept[:max_candidates]
    return kept