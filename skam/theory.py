"""Analytic results of the paper, for overlays and checks.

  P_eps(SNR, D)          Eq. 19                ML decoding error
  SNR_0                  Eq. 22, SI Eq. 34     N/L, times 2/pi after binarization
  per-overlap moments    SI Eqs. 19-20         HRR overlaps before binarization
  m0(L0, L)              Eq. 30, SI Eq. 37     cue overlap
  l_c(m_min)             Eq. 31                inverse of Eq. 30
  SNR out of basin       Eq. 9
  t0                     SI Eq. 89             dwell time, sequence of attractors
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr

C_OVERLAP = 0.65        # c in Eq. 8, as quoted in the paper


def p_epsilon(snr: float, D: int) -> float:
    """Eq. 19: int Dz [1 - (1 - H(z + sqrt(SNR)))^D], with H(x) = erfc(x/sqrt2)/2,
    i.e. 1 - H(-z - sqrt SNR)^D. Evaluated in log space for large D."""
    if snr < 0 or D < 1:
        raise ValueError(f"need snr >= 0 and D >= 1, got {snr}, {D}")
    s = np.sqrt(snr)

    def integrand(z: float) -> float:
        return float(np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
                     * -np.expm1(D * log_ndtr(z + s)))

    lo, hi = -s - 12.0, 12.0
    val, _ = quad(integrand, lo, hi, limit=400, points=[-s])
    # below lo the integrand is Dz to within e^-72; above hi the Gaussian mass is < 1e-32
    return float(val + np.exp(log_ndtr(lo)))


def snr0(N: int, L: int, binarized: bool) -> float:
    """Eq. 22 (N/L); SI Eq. 34 multiplies by 2/pi for a binarized structure."""
    return (2.0 / np.pi if binarized else 1.0) * N / L


def hrr_overlap_variance(N: int, L: int, kind: str) -> float:
    """SI Eq. 20: variance of a_d . a_hat_1 for the correct item ('correct'),
    another item of the structure ('in_structure'), or an outside item ('outside')."""
    if kind == "correct":
        return (L + 5) / N + 2 / N ** 2
    if kind == "in_structure":
        return (L + 3) / N + 1 / N ** 2
    if kind == "outside":
        return (L + 1) / N + 1 / N ** 2
    raise ValueError(f"unknown kind {kind!r}")


def m0_of_cue(L0: int | float, L: int | float) -> float:
    """Eq. 30: m0 = (2/pi) arctan( sqrt( (L0/L) / (1 - L0/L) ) )."""
    x = L0 / L
    if not 0 <= x <= 1:
        raise ValueError(f"L0/L={x} outside [0, 1]")
    if x == 1:
        return 1.0
    return float(2.0 / np.pi * np.arctan(np.sqrt(x / (1.0 - x))))


def lc_of_mmin(m_min: float) -> float:
    """Eq. 31: l_c = tan^2(pi m_min / 2) / (1 + tan^2(pi m_min / 2)) = sin^2(pi m_min / 2)."""
    return float(np.sin(np.pi * m_min / 2.0) ** 2)


def snr_out_of_basin(N: int, L: int, L0: int, f_alpha: float, c: float = C_OVERLAP) -> float:
    """Eq. 9: SNR = 8 c f(alpha)^2 / pi^3 * N L0 / L^2."""
    return 8.0 * c * f_alpha ** 2 / np.pi ** 3 * N * L0 / L ** 2


def ks_period(tau: float, lam: float) -> float:
    """SI Eq. 89: t0 = (tau / 2)(1 + 1/lambda)."""
    return tau / 2.0 * (1.0 + 1.0 / lam)
