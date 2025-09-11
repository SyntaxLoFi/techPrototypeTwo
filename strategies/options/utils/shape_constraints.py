"""
Shape constraint utilities for options pricing.

Projects a vector of European call prices C(K) to the closest (weighted least
squares) curve that is:
  • nonincreasing in strike K
  • convex in K (i.e., discrete slopes are nondecreasing)

These are the discrete static-arbitrage conditions (call-spread ≥ 0; butterfly ≥ 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import minimize

__all__ = ["project_call_curve_monotone_convex", "ProjectionInfo"]


@dataclass
class ProjectionInfo:
    success: bool
    message: str
    n_iter: Optional[int]
    obj_value: float


def _prepare_strikes_and_prices(
    strikes: np.ndarray,
    prices: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and sort inputs; merge duplicate strikes; broadcast & sanitize weights.

    Returns
    -------
    K : (m,) float64 strictly increasing, unique
    C : (m,) float64 prices aligned with K
    w : (m,) float64 nonnegative weights (sum(w) > 0 guaranteed)
    """
    K = np.asarray(strikes, dtype=float).reshape(-1)
    C = np.asarray(prices, dtype=float).reshape(-1)
    if K.size != C.size:
        raise ValueError(f"strikes and prices must have same length, got {K.size} and {C.size}")

    order = np.argsort(K)
    K = K[order]
    C = C[order]

    def _broadcast_w(raw, shape):
        if raw is None:
            return np.ones(shape, dtype=float)
        w_raw = np.asarray(raw, dtype=float)
        if w_raw.size == 1:
            return np.full(shape, float(w_raw), dtype=float)
        w_raw = w_raw.reshape(-1)
        if w_raw.size != shape[0]:
            try:
                w_raw = np.broadcast_to(w_raw, shape)
            except Exception as e:
                raise ValueError(f"weights cannot be broadcast to shape {shape}: {e}")
        return w_raw.astype(float)

    w_full = _broadcast_w(weights, K.shape)
    w_full = np.clip(w_full, 0.0, np.inf)  # guard ≥ 0

    # Merge duplicates via weight-averaged price, aggregate weights
    if np.any(np.diff(K) == 0):
        uniqK, idx = np.unique(K, return_inverse=True)
        C_agg = np.zeros_like(uniqK, dtype=float)
        w_agg = np.zeros_like(uniqK, dtype=float)
        for i in range(uniqK.size):
            m = (idx == i)
            wsum = float(np.sum(w_full[m]))
            if wsum > 0:
                C_agg[i] = float(np.sum(w_full[m] * C[m]) / wsum)
                w_agg[i] = wsum
            else:
                C_agg[i] = float(np.mean(C[m]))
                w_agg[i] = float(m.sum())
        K, C, w = uniqK, C_agg, w_agg
    else:
        w = w_full

    if not np.any(w > 0):
        w = np.ones_like(K, dtype=float)

    return K.astype(float), C.astype(float), w.astype(float)


def _build_linear_inequalities(K: np.ndarray):
    """
    Build A, b for inequality constraints g(C) = A C - b >= 0:

      1) Monotone:   C_i - C_{i+1} >= 0
      2) Convexity:  slope_i - slope_{i-1} >= 0, where
                     slope_i = (C_{i+1}-C_i)/(K_{i+1}-K_i)
    """
    n = K.size
    rows = []

    # Monotone nonincreasing
    for i in range(n - 1):
        r = np.zeros(n)
        r[i] = 1.0
        r[i + 1] = -1.0
        rows.append(r)

    # Convexity on nonuniform grid
    dK = np.diff(K)
    if np.any(dK <= 0):
        raise ValueError("strikes must be strictly increasing after deduplication")

    for i in range(1, n - 1):
        r = np.zeros(n)
        r[i + 1] += 1.0 / dK[i]
        r[i]     -= 1.0 / dK[i]
        r[i - 1] += 1.0 / dK[i - 1]
        r[i]     -= 1.0 / dK[i - 1]
        rows.append(r)

    A = np.vstack(rows) if rows else np.zeros((0, n))
    b = np.zeros(A.shape[0])
    return A, b


def project_call_curve_monotone_convex(
    strikes: np.ndarray,
    prices: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    lower_bound: float = 0.0,
    upper_bound: Optional[float] = None,
    tol: float = 1e-9,
    max_iter: int = 500,
    return_info: bool = False,
):
    """
    Weighted least-squares projection onto the cone:
      • nonincreasing in K
      • convex in K (nondecreasing discrete slopes)

    Parameters
    ----------
    strikes : (n,) array_like
        Strike vector (duplicates allowed; merged internally). Sorted ascending.
    prices : (n,) array_like
        Raw call prices C0(K).
    weights : scalar | (n,) array_like, optional
        Per-point nonnegative weights for the LS objective. Scalar is broadcast.
        Negatives are clipped to zero. If all weights are zero, uniform weights
        are used.
    lower_bound : float, default 0.0
        Elementwise lower bound (e.g., 0 for calls).
    upper_bound : float | None, default None
        Optional elementwise upper bound (e.g., forward cap). If None, no UB.
    tol : float, default 1e-9
        SLSQP ftol.
    max_iter : int, default 500
        SLSQP iteration cap.
    return_info : bool, default False
        If True, returns (projected, ProjectionInfo).

    Returns
    -------
    projected : (m,) ndarray
        Projected prices on the deduplicated strike grid.
    info : ProjectionInfo (optional)
    """
    K, C0, w = _prepare_strikes_and_prices(strikes, prices, weights)
    n = K.size

    # Bounds
    lb = np.full(n, float(lower_bound))
    if upper_bound is None:
        bounds = [(lb[i], None) for i in range(n)]
    else:
        ub = np.full(n, float(upper_bound))
        bounds = [(lb[i], ub[i]) for i in range(n)]

    A, b = _build_linear_inequalities(K)

    # Objective and gradient
    def obj(C):
        diff = C - C0
        return 0.5 * np.dot(w * diff, diff)

    def grad(C):
        return w * (C - C0)

    cons = []
    if A.size > 0:
        cons.append({"type": "ineq", "fun": lambda C: A @ C - b, "jac": lambda C: A})

    # Warm start: monotone clamp with lower bound; helps SLSQP robustness
    C_init = np.minimum.accumulate(np.maximum(C0, lb))

    res = minimize(
        fun=obj,
        x0=C_init,
        jac=grad,
        bounds=bounds,
        constraints=cons,
        method="SLSQP",
        options=dict(maxiter=int(max_iter), ftol=float(tol), disp=False),
    )

    if not res.success:
        proj = C_init
        info = ProjectionInfo(False, str(res.message), getattr(res, "nit", None), float(obj(proj)))
    else:
        proj = np.asarray(res.x, dtype=float)
        info = ProjectionInfo(True, str(res.message), getattr(res, "nit", None), float(obj(proj)))

    return (proj, info) if return_info else proj