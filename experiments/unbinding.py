"""Unbinding from structures before storage (Fig. 1b-c).

One trial: ``n_structures`` structures of L relations with distinct objects and
shared attributes, binarized (``binarized``), then relation L is unbound and
cleaned up against a Dictionary of D = d N items for each d in ``D_factors``.

Point keys: N, L, n_structures, D_factors (list of int), binarized (bool).
Outputs per d: error_D{d}, signal_D{d}, noise_D{d}, each (n_structures,).
"""
from __future__ import annotations

import numpy as np

from skam.trials import Point, TrialResult
from skam.vsa import HRR, Dictionary, ItemBank, Structures, binarize


KEYS = frozenset({"N", "L", "n_structures", "D_factors", "binarized"})


def trial(point: Point, seed: np.random.SeedSequence) -> TrialResult:
    N, L, B = point["N"], point["L"], point["n_structures"]
    s_bank, s_attr, s_read = seed.spawn(3)
    bank = ItemBank(N, B * L, s_bank)
    attrs = np.random.default_rng(s_attr).normal(0.0, 1.0 / np.sqrt(N), size=(L, N))
    st = Structures(bank, np.arange(B * L).reshape(B, L), attrs, HRR(asymmetric=False))
    S = st.encode([L])[L]
    if point["binarized"]:
        S = binarize(S)
    q = L - 1
    a_hat = st.unbind(S, q)
    true = st.object_ids[:, q]
    rng = np.random.default_rng(s_read)
    out: TrialResult = {}
    for d in point["D_factors"]:
        r = Dictionary(bank, B * L, d * N).readout(a_hat, true, rng)
        out[f"error_D{d}"], out[f"signal_D{d}"], out[f"noise_D{d}"] = r.error, r.signal, r.noise_sq
    return out
