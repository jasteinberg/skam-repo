"""Store, retrieve and decode structured memories (Figs. 3, 5; SI Figs. 1b, 3a, 8).

One trial is one network (Methods, Simulations): P = alpha N structures of L
relations with distinct objects and shared attributes b_l, binarized and stored
with ``rule``. Each structure is cued with its first L0 relations,
sigma(0) = sgn(S_hat_0) (Eq. 6), and evolved for T parallel updates. Relation L,
which is in none of the cues, is unbound from the retrieved state and cleaned
up against a Dictionary of D = D_factor N objects. The same read-out from the
stored pattern sigma itself is the "original structure".

Point keys: N, alpha, L, L0s (list), rule, T, D_factor, decode (bool).
Outputs, each (P,):
  m_L0{k}        overlap of the retrieved state with sigma (Eq. 29)
  mcue_L0{k}     overlap of the cue with sigma, m0 of Eq. 30
  error/signal/noise_L0{k} and error/signal/noise_original, if decode
"""
from __future__ import annotations

import numpy as np

from skam.measures import overlap
from skam.memory import learn, run_parallel
from skam.trials import Point, TrialResult
from skam.vsa import HRR, Dictionary, ItemBank, Structures, binarize


KEYS = frozenset({"N", "alpha", "L", "L0s", "rule", "T", "D_factor", "decode"})


def trial(point: Point, seed: np.random.SeedSequence) -> TrialResult:
    N, L, T = point["N"], point["L"], point["T"]
    L0s = list(point["L0s"])
    P = int(round(point["alpha"] * N))
    if any(not 1 <= k < L for k in L0s):
        raise ValueError(f"cue lengths {L0s} must lie in [1, L)")
    s_bank, s_attr, s_read = seed.spawn(3)
    bank = ItemBank(N, P * L, s_bank)
    attrs = np.random.default_rng(s_attr).normal(0.0, 1.0 / np.sqrt(N), size=(L, N))
    st = Structures(bank, np.arange(P * L).reshape(P, L), attrs, HRR(asymmetric=False))
    enc = st.encode([L] + L0s)
    sigma = binarize(enc[L])
    J = learn(point["rule"], sigma)

    out: TrialResult = {}
    states = {"original": sigma}
    for k in L0s:
        cue = binarize(enc[k])
        final = run_parallel(J, cue, T).final
        out[f"m_L0{k}"] = overlap(final, sigma)
        out[f"mcue_L0{k}"] = overlap(cue, sigma)
        states[f"L0{k}"] = final
    if point["decode"]:
        q = L - 1
        dic = Dictionary(bank, P * L, point["D_factor"] * N)
        rng = np.random.default_rng(s_read)
        for name, S in states.items():
            r = dic.readout(st.unbind(S, q), st.object_ids[:, q], rng)
            out[f"error_{name}"], out[f"signal_{name}"], out[f"noise_{name}"] = (
                r.error, r.signal, r.noise_sq)
    return out
