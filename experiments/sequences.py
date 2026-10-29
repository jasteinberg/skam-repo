"""Temporal sequences stored as single attractors (Fig. 6b, 6e, 6f).

One trial is one network storing P sequences of L events (Eqs. 10-11,
asymmetric HRR), binarized and stored with ``rule``. Sequence mu+1 shares a
contiguous run of ``n_shared`` events with sequence mu, at the same positions
in both. A run of n events gives n - 1 shared relations g(a_{l+1}, a_l), which
is what correlates the stored patterns. The run starts at ``shared_start``
(1-based) or, if None, uniformly in 3..L - n_shared + 1 for each pair;
positions 1-2 are never shared, so every cue (a_1, a_2) is unique.

Retrieval cue: the first relation, sigma(0) = sgn(g(a_2, a_1)) (main text).
After T parallel updates the sequence is unfolded from a_1 by repeated
unbind -> clean-up, feeding each cleaned-up item forward (Fig. 6a).

Dictionary: D = D_factor N items as in Methods (D = L_max N), the same for
every n_shared; at high SNR P_eps grows linearly in D (Eq. 21). The P L slot
items are explicit (an item displaced by sharing stays as a familiar item in
no stored sequence); the rest are handled exactly as in skam.vsa.

Point keys: N, P, L, n_shared, shared_start (1-based int or None), rule, T, D_factor.
Outputs:
  correct  (P, L) bool   decoded item equals the true one (column 0 is a_1)
  own      (P, L)        a_hat_l . a_l^mu                    (column 0 zero)
  next_seq (P, L)        a_hat_l . a_l^{mu+1 mod P}          (column 0 zero)
  m        (P,)          overlap of the retrieved state with the stored pattern
"""
from __future__ import annotations

import numpy as np

from skam.measures import overlap
from skam.memory import learn, run_parallel
from skam.trials import Point, TrialResult
from skam.vsa import HRR, Dictionary, ItemBank, Sequences, binarize


KEYS = frozenset({"N", "P", "L", "n_shared", "shared_start", "rule", "T", "D_factor"})


def item_ids(P: int, L: int, n_shared: int, start: int | None,
             rng: np.random.Generator) -> np.ndarray:
    """(P, L) ids into a bank of P L items; slot (mu, l) owns id mu L + l unless
    it lies in the run shared with sequence mu - 1."""
    if not 0 <= n_shared <= L - 2:
        raise ValueError(f"n_shared={n_shared} must lie in [0, {L - 2}]")
    if start is not None and not 3 <= start <= L - n_shared + 1:
        raise ValueError(f"shared_start={start} outside 3..{L - n_shared + 1}")
    ids = np.arange(P * L, dtype=np.int64).reshape(P, L)
    for mu in range(1, P):
        if n_shared == 0:
            continue
        k = (start if start is not None else int(rng.integers(3, L - n_shared + 2))) - 1
        ids[mu, k:k + n_shared] = ids[mu - 1, k:k + n_shared]
    return ids


def trial(point: Point, seed: np.random.SeedSequence) -> TrialResult:
    N, P, L, T = point["N"], point["P"], point["L"], point["T"]
    s_bank, s_ids, s_read = seed.spawn(3)
    ids = item_ids(P, L, point["n_shared"], point["shared_start"],
                   np.random.default_rng(s_ids))
    n_items = P * L
    bank = ItemBank(N, n_items, s_bank)
    seqs = Sequences(bank, ids, HRR(asymmetric=True))
    sigma = binarize(seqs.encode(L - 1))
    cue = binarize(seqs.encode(1))
    J = learn(point["rule"], sigma)
    final = run_parallel(J, cue, T).final
    dic = Dictionary(bank, n_items, point["D_factor"] * N)
    decoded, a_hats = seqs.unfold(final, dic, np.random.default_rng(s_read))
    items = bank.get(ids)                                         # (P, L, N)
    own = np.einsum("pln,pln->pl", a_hats, items)
    other = np.einsum("pln,pln->pl", a_hats, np.roll(items, -1, axis=0))
    return {"correct": decoded == ids, "own": own, "next_seq": other,
            "m": overlap(final, sigma)}
