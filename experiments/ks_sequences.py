"""Two sequences of attractors sharing one pattern (Fig. 6d; SI Sec. 4).

One trial: 2L - 1 random +/-1 patterns forming sequence 1 (xi^1..xi^L) and
sequence 2, which equals sequence 1 at position ``shared`` (1-based) and is
otherwise distinct. J^(1) from ``rule`` over the distinct patterns, J^(2) from
both sequences (SI Eqs. 85-86). Start at sigma(0) = xi^1 and run T parallel
steps of SI Eq. 87 with window tau and strength lam.

Point keys: N, L, shared, tau, lam, T, rule.
Outputs: m (T+1, 2L-1), overlap with every pattern; seq (2, L) pattern indices.
"""
from __future__ import annotations

import numpy as np

from skam import ks
from skam.trials import Point, TrialResult


KEYS = frozenset({"N", "L", "shared", "tau", "lam", "T", "rule"})


def trial(point: Point, seed: np.random.SeedSequence) -> TrialResult:
    N, L, k = point["N"], point["L"], point["shared"] - 1
    if not 0 <= k < L:
        raise ValueError(f"shared position {point['shared']} outside 1..{L}")
    rng = np.random.default_rng(seed)
    patterns = rng.choice(np.array([-1.0, 1.0]), size=(2 * L - 1, N))
    seq1 = np.arange(L)
    seq2 = np.concatenate([np.arange(L, L + k), [k], np.arange(L + k, 2 * L - 1)])
    seqs = np.stack([seq1, seq2])
    J1, J2 = ks.build(patterns, seqs, point["lam"], point["rule"])
    hist = ks.run(J1, J2, patterns[:1], point["T"], point["tau"])        # (T+1, 1, N)
    return {"m": hist[:, 0, :] @ patterns.T / N, "seq": seqs}
