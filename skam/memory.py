"""Binary recurrent memory network (Methods, Memory network; SI Sec. 3).

Neurons are +/-1. The dynamics (Eq. 23) is

    sigma_i(t) = sgn( sum_j J_ij sigma_j(t-1) )

with parallel or serial (random sequential) updates. A local field h_i = 0
leaves sigma_i unchanged. Serial updates at finite inverse temperature beta
use the Metropolis acceptance of SI Eq. 66,

    p(sigma_i -> -sigma_i) = min(1, exp(-2 beta h_i sigma_i)).

One time step of serial dynamics is one sweep over all N sites in a random
order, drawn afresh each sweep and shared by the chains of a batch (each
chain's marginal dynamics is exact; chains in one batch share the order).

Learning rules, all with J_ii = 0:
  pseudo-inverse  J = (1/N) sigma^T C^{-1} sigma,  C = sigma sigma^T / N   (Eqs. 24-25)
  Hebb            J = (1/N) sigma^T sigma                                 (Eq. 26)
  Storkey         recurrence of Eqs. 27-28, as printed in Methods

Pseudo-inverse and Hebb couplings are rank P plus a diagonal, so they are kept
in factored form: the field of a batch of B states costs O(B P N) and no N x N
matrix is formed. Storkey couplings are dense.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_solve, lapack

from . import backend

FloatArray = NDArray[np.float64]

RULES = ("pseudo_inverse", "hebb", "storkey")


class Couplings(ABC):
    """Symmetric synaptic matrix J with zero diagonal."""
    N: int

    @abstractmethod
    def field(self, S: FloatArray) -> FloatArray:
        """Local fields h = J s for each row s of S (B, N)."""

    @abstractmethod
    def dense(self) -> FloatArray:
        """J as an explicit (N, N) array."""


class LowRankCouplings(Couplings):
    """J = (1/N) V^T W - diag(d), with V, W of shape (P, N) and d_i = (V^T W)_ii / N."""

    def __init__(self, V: FloatArray, W: FloatArray):
        if V.shape != W.shape:
            raise ValueError(f"V {V.shape} and W {W.shape} differ")
        self.V = V
        self.W = W
        self.N = V.shape[1]
        self.diag = np.einsum("pi,pi->i", V, W) / self.N

    def field(self, S: FloatArray) -> FloatArray:
        return (S @ self.V.T) @ self.W / self.N - S * self.diag

    def dense(self) -> FloatArray:
        J = self.V.T @ self.W / self.N
        np.fill_diagonal(J, 0.0)
        return J


class DenseCouplings(Couplings):
    def __init__(self, J: FloatArray):
        if J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError(f"J must be square, got {J.shape}")
        if np.any(np.diag(J) != 0):
            raise ValueError("J must have zero diagonal")
        self.J = J
        self.N = J.shape[0]

    def field(self, S: FloatArray) -> FloatArray:
        return S @ self.J.T

    def dense(self) -> FloatArray:
        return self.J


def _check_patterns(patterns: FloatArray) -> FloatArray:
    patterns = np.asarray(patterns, dtype=np.float64)
    if patterns.ndim != 2:
        raise ValueError(f"patterns must be (P, N), got {patterns.shape}")
    if not np.all(np.abs(patterns) == 1):
        raise ValueError("patterns must be +/-1")
    return patterns


def pseudo_inverse(patterns: FloatArray) -> LowRankCouplings:
    """Eqs. 24-25. C is symmetric positive definite for linearly independent
    patterns, so C^{-1} sigma is computed by Cholesky in O(P^3 + P^2 N); the
    LAPACK estimate of cond(C) costs O(P^2)."""
    sig = _check_patterns(patterns)
    N = sig.shape[1]
    C = sig @ sig.T / N
    factor, info = lapack.dpotrf(C, lower=False)
    if info != 0:
        raise np.linalg.LinAlgError("pattern overlap matrix C is not positive definite")
    rcond, info = lapack.dpocon(factor, np.abs(C).sum(axis=0).max())
    if info != 0 or rcond < 1e-10:
        raise np.linalg.LinAlgError(f"pattern overlap matrix C is singular (rcond={rcond:.3g})")
    return LowRankCouplings(V=sig, W=cho_solve((factor, False), sig))


def hebb(patterns: FloatArray) -> LowRankCouplings:
    """Eq. 26."""
    sig = _check_patterns(patterns)
    return LowRankCouplings(V=sig, W=sig)


def storkey(patterns: FloatArray) -> DenseCouplings:
    """Eqs. 27-28 as printed in Methods:

        J^mu = (N+1)/(N-1) J^{mu-1} + (sigma sigma^T - I)/(N-1)
               - (sigma h^T + h sigma^T)/(N-1),     h = J^{mu-1} sigma,   J^mu_ii = 0
    """
    sig = _check_patterns(patterns)
    N = sig.shape[1]
    J = np.zeros((N, N))
    for s in sig:
        h = J @ s
        J *= (N + 1) / (N - 1)
        J += (np.outer(s, s) - np.outer(s, h) - np.outer(h, s)) / (N - 1)
        np.fill_diagonal(J, 0.0)        # the -delta_ij term only touches the diagonal
    return DenseCouplings(J)


def learn(rule: str, patterns: FloatArray) -> Couplings:
    if rule == "pseudo_inverse":
        return pseudo_inverse(patterns)
    if rule == "hebb":
        return hebb(patterns)
    if rule == "storkey":
        return storkey(patterns)
    raise ValueError(f"unknown learning rule {rule!r}; choose from {RULES}")


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Trajectory:
    """final: states after T steps (B, N).
    overlap: m(t) = (1/N) sigma(t) . target for t = 0..T, shape (T+1, B), if
    targets were given, else None."""
    final: FloatArray
    overlap: FloatArray | None


def _overlap(S: FloatArray, targets: FloatArray | None) -> FloatArray | None:
    return None if targets is None else np.einsum("bi,bi->b", S, targets) / S.shape[1]


def run_parallel(J: Couplings, S0: FloatArray, T: int,
                 targets: FloatArray | None = None) -> Trajectory:
    """T parallel updates of Eq. 23. Chains that reach a fixed point are frozen
    (their later updates would not change them). Runs on the active backend."""
    if backend.name() == "torch":
        return _run_parallel_torch(J, S0, T, targets)
    S = np.array(S0, dtype=np.float64)
    m = [_overlap(S, targets)]
    active = np.ones(S.shape[0], dtype=bool)
    for _ in range(T):
        if active.any():
            Sa = S[active]
            h = J.field(Sa)
            new = np.where(h > 0, 1.0, np.where(h < 0, -1.0, Sa))
            changed = np.any(new != Sa, axis=1)
            S[active] = new
            idx = np.flatnonzero(active)
            active[idx[~changed]] = False
        m.append(_overlap(S, targets))
    return Trajectory(final=S, overlap=None if targets is None else np.array(m))


def _run_parallel_torch(J: Couplings, S0: FloatArray, T: int,
                        targets: FloatArray | None) -> Trajectory:
    """run_parallel in float32 on the torch device; same update and freezing rule."""
    torch = backend.torch_module()
    dev, f32 = backend.device(), torch.float32

    def t(x: FloatArray) -> Any:
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=dev)

    if isinstance(J, LowRankCouplings):
        V, W, d, N = t(J.V), t(J.W), t(J.diag), J.N
        def field(S: Any) -> Any:
            return (S @ V.T) @ W / N - S * d
    else:
        Jt = t(J.dense())
        def field(S: Any) -> Any:
            return S @ Jt.T
    S = t(S0)
    tg = None if targets is None else t(targets)

    def ov() -> FloatArray | None:
        return None if tg is None else ((S * tg).sum(dim=1) / S.shape[1]).cpu().numpy().astype(np.float64)

    m = [ov()]
    active = torch.ones(S.shape[0], dtype=torch.bool, device=dev)
    one = torch.ones((), dtype=f32, device=dev)
    for _ in range(T):
        if bool(active.any()):
            idx = torch.nonzero(active).squeeze(1)
            Sa = S[idx]
            h = field(Sa)
            new = torch.where(h > 0, one, torch.where(h < 0, -one, Sa))
            changed = (new != Sa).any(dim=1)
            S[idx] = new
            active[idx[~changed]] = False
        m.append(ov())
    final = S.cpu().numpy().astype(np.float64)
    return Trajectory(final=final, overlap=None if targets is None else np.array(m))


def run_serial(J: Couplings, S0: FloatArray, T: int, beta: float | None,
               rng: np.random.Generator, targets: FloatArray | None = None) -> Trajectory:
    """T sweeps of random sequential updates. beta=None is the zero-noise
    limit (Eq. 64); finite beta uses Metropolis acceptance (SI Eq. 66)."""
    Jd = J.dense()
    S = np.array(S0, dtype=np.float64)
    B, N = S.shape
    h = S @ Jd.T
    m = [_overlap(S, targets)]
    for _ in range(T):
        for i in rng.permutation(N):
            hi, si = h[:, i], S[:, i]
            if beta is None:
                flip = hi * si < 0
            else:
                p = np.exp(np.minimum(-2.0 * beta * hi * si, 0.0))
                flip = rng.random(B) < p
            if flip.any():
                rows = np.flatnonzero(flip)
                S[rows, i] = -S[rows, i]
                h[rows] += 2.0 * S[rows, i][:, None] * Jd[:, i][None, :]
        m.append(_overlap(S, targets))
    return Trajectory(final=S, overlap=None if targets is None else np.array(m))


def retrieve(J: Couplings, S0: FloatArray, T: int, update: str, *,
             beta: float | None = None, rng: np.random.Generator | None = None,
             targets: FloatArray | None = None) -> Trajectory:
    if update == "parallel":
        if beta is not None:
            raise ValueError("finite-temperature dynamics is defined for serial updates only")
        return run_parallel(J, S0, T, targets)
    if update == "serial":
        if rng is None:
            raise ValueError("serial updates need an rng for the site order")
        return run_serial(J, S0, T, beta, rng, targets)
    raise ValueError(f"unknown update {update!r}; choose 'parallel' or 'serial'")
