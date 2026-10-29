"""Order parameters and summary statistics, one implementation per quantity.

  m        overlap of a state with a pattern                   (Eq. 29, SI Eq. 60)
  r        (1/alpha) sum_{nu != mu} m_nu^2                      (SI Eq. 61)
  m^0      overlap of the final state with the initial state    (SI Eq. 62)
  SNR      <a_hat . a_d>^2 / <(a_hat . a_d')^2>                 (Eq. 20)
  p1, m*, sigma_m^2   shape of p(m)                             (Eq. 32)
  m_min    smallest m0 that converges to the memory on average  (main text, Methods)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def overlap(S: FloatArray, targets: FloatArray) -> FloatArray:
    """Row-wise m = (1/N) s . target, shapes (B, N) -> (B,)."""
    if S.shape != targets.shape:
        raise ValueError(f"shape mismatch {S.shape} vs {targets.shape}")
    return np.einsum("bi,bi->b", S, targets) / S.shape[1]


def order_r(S: FloatArray, patterns: FloatArray, cued: NDArray[np.int64]) -> FloatArray:
    """r = (1/alpha) sum_{nu != mu} m_nu^2 for each state, mu = cued pattern (SI Eq. 61)."""
    P, N = patterns.shape
    m = S @ patterns.T / N                                    # (B, P)
    m[np.arange(S.shape[0]), cued] = 0.0
    return np.sum(m * m, axis=1) / (P / N)


def snr(signal: FloatArray, noise_sq: FloatArray) -> float:
    """Eq. 20 over a population of estimators: mean(signal)^2 / mean(noise_sq)."""
    signal, noise_sq = np.asarray(signal), np.asarray(noise_sq)
    if signal.size == 0 or signal.shape != noise_sq.shape:
        raise ValueError("snr needs equal-length, non-empty signal and noise arrays")
    return float(np.mean(signal) ** 2 / np.mean(noise_sq))


def flip_to_overlap(patterns: FloatArray, m0: float, rng: np.random.Generator) -> FloatArray:
    """Copy of each pattern with exactly round(N (1 - m0) / 2) spins flipped,
    so that its overlap with the pattern is m0 up to 1/N rounding."""
    P, N = patterns.shape
    if not 0.0 <= m0 <= 1.0:
        raise ValueError(f"m0={m0} outside [0, 1]")
    k = int(round(N * (1.0 - m0) / 2.0))
    S = patterns.copy()
    if k:
        idx = np.argsort(rng.random((P, N)), axis=1)[:, :k]
        rows = np.arange(P)[:, None]
        S[rows, idx] *= -1.0
    return S


@dataclass(frozen=True)
class OverlapShape:
    """Shape of p(m) = (1 - p1) p_{m<1}(m) + p1 delta(m - 1) (Eq. 32).
    m_star and var_m are NaN when every state reached m = 1 (no lower mode)."""
    p1: float
    m_star: float
    var_m: float


def overlap_shape(m: FloatArray, N: int) -> OverlapShape:
    m = np.asarray(m, dtype=np.float64).ravel()
    if m.size == 0:
        raise ValueError("empty overlap sample")
    perfect = m > 1.0 - 0.5 / N                 # m is a multiple of 2/N
    low = m[~perfect]
    return OverlapShape(p1=float(perfect.mean()),
                        m_star=float(low.mean()) if low.size else float("nan"),
                        var_m=float(low.var()) if low.size else float("nan"))


def first_crossing(x: FloatArray, y: FloatArray, level: float) -> float:
    """Smallest x at which y reaches ``level``, by linear interpolation between
    grid points. Raises if y never reaches it on the grid."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    above = np.flatnonzero(y >= level)
    if above.size == 0:
        raise ValueError(f"y never reaches {level} on the grid")
    k = above[0]
    if k == 0:
        return float(x[0])
    x0, x1, y0, y1 = x[k - 1], x[k], y[k - 1], y[k]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


M_CONVERGED = 0.99


def convergence_threshold(x: FloatArray, mean_m: FloatArray) -> float:
    """Smallest x (interpolated) at which the mean final overlap with the
    memory reaches M_CONVERGED: the paper's "minimal overlap ... required for
    convergence to the correct fixed point on average". Used for m_min
    (x = m0) and l_c (x = L0/L)."""
    return first_crossing(x, mean_m, M_CONVERGED)
