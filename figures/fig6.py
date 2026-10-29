"""Fig. 6b, d, e, f: sequences.

(b) Accuracy vs position l for L in {10, 14, 18}; N = 2500, alpha = 0.05, 100 trials.
(d) Sequence-of-attractors model, two sequences sharing one pattern;
    N = 1000, tau = 8, lambda = 2.5, 1000 trials. <m> with every pattern vs t.
(e) Two holistic sequences sharing one event; N = 1000, 10,000 trials.
    a_hat_l . a_l with the first sequence's item and with the second
    sequence's item at the same position, normalized by <a_hat_2 . a_2>.
(f) Accuracy vs l for L = 18 with a contiguous run of L_s in {0, 1, 2, 4}
    events shared between neighbouring sequences; N = 2500, alpha = 0.05, 100 trials.

Not fixed by the paper and chosen here: (d) sequence length 15, shared at
position 7, J^(1) pseudo-inverse, 100 steps; (e) L = 14, shared position 7
(read off the panel); learning rule pseudo-inverse throughout. Dictionary
D = 30N as in Methods.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from skam import trials
from skam.trials import Job

from .common import Ctx, D_FACTOR, Figure, SERIES, T_RETRIEVAL, series

SCALES: dict[str, dict[str, Any]] = {
    "paper": {"trials_bf": 100, "trials_d": 1000, "trials_e": 10000},
    # 6e: one trial is two sequences; 100 are needed to see a ~1%/step error rate
    "quick": {"trials_bf": 10, "trials_d": 10, "trials_e": 100},
}
N_BF, ALPHA_BF = 2500, 0.05
LS_B = [10, 14, 18]
LSS_F = [0, 1, 2, 4]
KS: dict[str, Any] = {"N": 1000, "L": 15, "shared": 7, "tau": 8, "lam": 2.5, "T": 100, "rule": "pseudo_inverse"}
E: dict[str, int] = {"N": 1000, "P": 2, "L": 14, "shared": 7}


def seq_job(N: int, P: int, L: int, n_shared: int, start: int | None, n: int) -> Job:
    return Job("sequences", {"N": int(N), "P": int(P), "L": int(L), "n_shared": int(n_shared),
                             "shared_start": start, "rule": "pseudo_inverse", "D_factor": D_FACTOR,
                             "T": T_RETRIEVAL}, n)


def _bf(L: int, Ls: int, scale: str) -> Job:
    return seq_job(N_BF, int(round(ALPHA_BF * N_BF)), L, Ls, None, SCALES[scale]["trials_bf"])


def _e(scale: str) -> Job:
    return seq_job(E["N"], E["P"], E["L"], 1, E["shared"], SCALES[scale]["trials_e"])


def _d(scale: str) -> Job:
    return Job("ks_sequences", dict(KS), SCALES[scale]["trials_d"])


def jobs(scale: str) -> list[Job]:
    return ([_bf(L, 0, scale) for L in LS_B] + [_bf(18, k, scale) for k in LSS_F if k]
            + [_d(scale), _e(scale)])


def _accuracy(tr: list[trials.TrialResult]) -> np.ndarray:
    return trials.stack(tr, "correct").mean(axis=0)            # (L,)


def plot(ctx: Ctx) -> dict[str, Figure]:
    fig, ax = plt.subplots(2, 2, figsize=(6.8, 5.4))
    b, d, e, f = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

    for i, L in enumerate(LS_B):
        acc = _accuracy(ctx.load(_bf(L, 0, ctx.scale)))
        b.plot(np.arange(2, L + 1), acc[1:], label=f"$L={L}$", **series(i))
    b.set(xlabel=r"$\ell$", ylabel="Accuracy", title=f"(b) $N={N_BF}$, $\\alpha={ALPHA_BF}$")
    b.legend()

    trd = ctx.load(_d(ctx.scale))
    m = np.mean([t["m"] for t in trd], axis=0)                 # (T+1, 2L-1)
    seq = trd[0]["seq"]
    t_axis = np.arange(m.shape[0])
    shared = int(seq[0, KS["shared"] - 1])
    for p in range(m.shape[1]):
        in1, in2 = p in seq[0], p in seq[1]
        color = SERIES[2] if (in1 and in2) else SERIES[1] if in1 else SERIES[0]
        d.plot(t_axis, m[:, p], color=color, lw=1.0, marker=".", ms=2)
    d.plot([], [], color=SERIES[1], label="First sequence")
    d.plot([], [], color=SERIES[0], label="Second sequence")
    d.plot([], [], color=SERIES[2], label=f"Shared (pattern {shared + 1})")
    d.set(xlabel="$t$", ylabel=r"$\langle m\rangle$",
          title=f"(d) $\\tau={KS['tau']}$, $\\lambda={KS['lam']}$")
    d.legend(loc="upper right")

    tre = ctx.load(_e(ctx.scale))
    own = trials.stack(tre, "own")[:: E["P"]]                    # first sequence of each trial
    nxt = trials.stack(tre, "next_seq")[:: E["P"]]
    norm = own[:, 1].mean()
    ls = np.arange(2, E["L"] + 1)
    e.plot(ls, own[:, 1:].mean(axis=0) / norm, label="First sequence", **series(1))
    e.plot(ls, nxt[:, 1:].mean(axis=0) / norm, ls="none", marker="x",
           color=SERIES[0], label="Second sequence")
    e.set(xlabel=r"$\ell$", ylabel=r"Overlap with $\hat a_\ell$", title=f"(e) $N={E['N']}$")
    e.legend()

    for i, k in enumerate(LSS_F):
        acc = _accuracy(ctx.load(_bf(18, k, ctx.scale)))
        f.plot(np.arange(2, 19), acc[1:], label=f"$L_s={k}$", **series(i))
    f.set(xlabel=r"$\ell$", ylabel="Accuracy", title="(f) $L=18$")
    f.legend()
    fig.tight_layout()
    return {"fig6": fig}
