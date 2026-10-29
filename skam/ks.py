"""Temporal sequences as sequences of attractors (SI Sec. 4, Fig. 6c-d).

The comparison model stores every pattern of every sequence as its own fixed
point through a symmetric J^(1) and chains them with an asymmetric term

    J^(2)_ij = (lambda / N) sum_q sum_l xi_i^{q,l+1} xi_j^{q,l},  i != j   (SI Eqs. 85-86)

acting on a delayed average of the state (SI Eqs. 87-88),

    sigma_i(t) = sgn( sum_j J^(1)_ij sigma_j(t-1) + sum_j J^(2)_ij sigmabar_j(t-1) ).

Kernel. SI Eq. 88 prints w(t) = (1/tau) Theta(t - tau), whose integral over
the past diverges. We use the sliding window w(t) = (1/tau) Theta(t) Theta(tau - t),
in discrete time sigmabar(t) = (1/tau) sum_{s=0}^{tau-1} sigma(t - s) with
sigma(t < 0) = 0. This window gives the dwell time t0 = (tau/2)(1 + 1/lambda)
of SI Eq. 89: entering pattern l at time 0, the field toward l+1 is
lambda t / tau and the field holding l is 1 + lambda (tau - t)/tau; they cross
at t = tau (1 + lambda) / (2 lambda).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .memory import Couplings, learn

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def transition_couplings(patterns: FloatArray, sequences: IntArray, lam: float) -> FloatArray:
    """J^(2) of SI Eqs. 85-86. ``sequences`` (Q, L) indexes rows of ``patterns``."""
    N = patterns.shape[1]
    nxt = patterns[sequences[:, 1:]].reshape(-1, N)
    prv = patterns[sequences[:, :-1]].reshape(-1, N)
    J2 = lam / N * nxt.T @ prv
    np.fill_diagonal(J2, 0.0)
    return J2


def run(J1: Couplings, J2: FloatArray, sigma0: FloatArray, T: int, tau: int) -> FloatArray:
    """Parallel dynamics of SI Eq. 87 from sigma(0) = sigma0 (B, N).
    Returns the states sigma(0..T), shape (T+1, B, N)."""
    if tau < 1:
        raise ValueError("tau must be >= 1")
    B, N = sigma0.shape
    hist = np.zeros((T + 1, B, N))
    hist[0] = sigma0
    for t in range(1, T + 1):
        window = hist[max(0, t - tau):t]                   # sigma(t-tau .. t-1)
        bar = window.sum(axis=0) / tau
        h = J1.field(hist[t - 1]) + bar @ J2.T
        prev = hist[t - 1]
        hist[t] = np.where(h > 0, 1.0, np.where(h < 0, -1.0, prev))
    return hist


def build(patterns: FloatArray, sequences: IntArray, lam: float, rule: str
          ) -> tuple[Couplings, FloatArray]:
    """J^(1) from ``rule`` over all distinct patterns, J^(2) from the sequences."""
    return learn(rule, patterns), transition_couplings(patterns, sequences, lam)
