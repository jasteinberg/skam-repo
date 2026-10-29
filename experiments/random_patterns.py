"""Retrieval of random +/-1 patterns from cues of set overlap m0
(Fig. 4b; SI Figs. 1a, 2, 3b, 4, 5, 6, 7).

One trial is one network storing P = alpha N random patterns with ``rule``.
For each m0 in ``m0s`` the first ``n_probe`` patterns are cued with exactly
round(N (1 - m0) / 2) spins flipped and evolved for T updates (``update``,
``beta``). Patterns are i.i.d., so probing the first n_probe is a uniform
sample. The reference fixed point of a pattern is the state the same
dynamics reaches from the pattern itself; it equals the pattern whenever the
pattern is a fixed point (always for the pseudo-inverse rule at alpha < 1) and
differs from it for Hebb or Storkey near capacity.

Point keys: N, alpha, rule, update, beta (None or float), T, m0s (list),
n_probe, trajectory (bool), order_params (bool).
Outputs:
  m       (n_m0, n_probe)        final overlap with the cued pattern
  m_ref   (n_m0, n_probe)        final overlap with the pattern's reference fixed point
  m_t     (T+1, n_m0, n_probe)   m(t), if trajectory
  m_init  (n_m0, n_probe)        final overlap with the initial state (SI Eq. 62), if order_params
  r       (n_m0, n_probe)        SI Eq. 61, if order_params
"""
from __future__ import annotations

import numpy as np

from skam.measures import flip_to_overlap, order_r, overlap
from skam.memory import learn, retrieve
from skam.trials import Point, TrialResult


KEYS = frozenset({"N", "alpha", "rule", "update", "beta", "T", "m0s", "n_probe", "trajectory", "order_params"})


def trial(point: Point, seed: np.random.SeedSequence) -> TrialResult:
    N, T = point["N"], point["T"]
    P = int(round(point["alpha"] * N))
    m0s = list(point["m0s"])
    n = min(point["n_probe"], P)
    rng = np.random.default_rng(seed)
    patterns = rng.choice(np.array([-1.0, 1.0]), size=(P, N))
    J = learn(point["rule"], patterns)
    targets = np.tile(patterns[:n], (len(m0s), 1))
    S0 = np.concatenate([flip_to_overlap(patterns[:n], m0, rng) for m0 in m0s])
    traj = retrieve(J, S0, T, point["update"], beta=point["beta"], rng=rng,
                    targets=targets if point["trajectory"] else None)
    ref = retrieve(J, patterns[:n], T, point["update"], beta=point["beta"], rng=rng).final
    shape = (len(m0s), n)
    out: TrialResult = {"m": overlap(traj.final, targets).reshape(shape),
                        "m_ref": overlap(traj.final, np.tile(ref, (len(m0s), 1))).reshape(shape)}
    if traj.overlap is not None:
        out["m_t"] = traj.overlap.reshape((T + 1,) + shape)
    if point["order_params"]:
        cued = np.tile(np.arange(n), len(m0s))
        out["m_init"] = overlap(traj.final, S0).reshape(shape)
        out["r"] = order_r(traj.final, patterns, cued).reshape(shape)
    return out
