"""Fitts' law primitives.

    MT = a + b * ID,      ID = log2(D / W + 1)      (Shannon formulation, MacKenzie 1992)

``a`` is in seconds, ``b`` in seconds per bit, throughput in bits per second.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def index_of_difficulty(d: np.ndarray | float, w: np.ndarray | float) -> np.ndarray:
    return np.log2(np.asarray(d, dtype=float) / np.asarray(w, dtype=float) + 1.0)


@dataclass
class FittsFit:
    a: float          # intercept, s
    b: float          # slope, s / bit
    r2: float         # coefficient of determination
    resid: np.ndarray  # MT - (a + b * ID), s
    n: int

    def predict(self, id_bits: np.ndarray | float) -> np.ndarray:
        return self.a + self.b * np.asarray(id_bits, dtype=float)


def fit_fitts(id_bits: np.ndarray, mt: np.ndarray) -> FittsFit:
    """Ordinary least squares of movement time on index of difficulty.

    Degenerate inputs (fewer than 3 trials or no spread in ID) yield NaN
    coefficients rather than raising: a session can legitimately consist of
    identical targets, and the caller decides what that means.
    """
    id_bits = np.asarray(id_bits, dtype=float)
    mt = np.asarray(mt, dtype=float)
    n = len(mt)
    if n < 3 or np.ptp(id_bits) < 1e-9:
        return FittsFit(np.nan, np.nan, np.nan, mt - np.mean(mt) if n else mt, n)
    x = id_bits - id_bits.mean()
    b = float(np.dot(x, mt) / np.dot(x, x))
    a = float(mt.mean() - b * id_bits.mean())
    resid = mt - (a + b * id_bits)
    ss_tot = float(np.sum((mt - mt.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 1e-12 else np.nan
    return FittsFit(a, b, r2, resid, n)


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares with an intercept column prepended. Returns (coef, r2)."""
    X = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return coef, r2


def throughput(id_bits: np.ndarray, mt: np.ndarray) -> float:
    """Mean-of-ratios throughput (bits/s), the ISO 9241-9 convention."""
    mt = np.asarray(mt, dtype=float)
    ok = mt > 0
    if not ok.any():
        return np.inf
    return float(np.mean(np.asarray(id_bits, dtype=float)[ok] / mt[ok]))


def effective_width(endpoint_along: np.ndarray) -> float:
    """We = 4.133 * SD of the endpoint coordinate along the task axis."""
    return float(4.133 * np.std(endpoint_along, ddof=1)) if len(endpoint_along) > 1 else np.nan
