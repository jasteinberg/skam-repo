"""Fig. 1b-c: unbinding error and inverse SNR vs L, before storage.

Caption: (b) P_eps vs L for N = 1000, D in {30N, 40N, 50N}; (c) SNR^-1 vs L for
N in {500, ..., 8000}, D = 30N; averages over 10,000 structures.
Structures are binarized: the published SNR^-1 at N = 500, L = 30 is 0.095,
matching (pi/2) L/N = 0.094 (SI Eq. 34) and not L/N = 0.06.
Dashed lines: Eq. 19 with SNR = (2/pi) N/L (b) and SI Eq. 34 (c).
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from skam import measures, theory, trials
from skam.trials import Job

from .common import Ctx, Figure, THEORY, series

SCALES: dict[str, dict[str, Any]] = {
    "paper": {"Nb": 1000, "Ds": [30, 40, 50], "Nc": [500, 1000, 2000, 4000, 8000],
              "Ls": list(range(2, 31)), "structures": 10000, "batch": 500},
    "quick": {"Nb": 1000, "Ds": [30, 40, 50], "Nc": [500, 1000, 2000],
              "Ls": list(range(2, 31, 2)), "structures": 2000, "batch": 200},
}


def _job(N: int, L: int, Ds: list[int], s: dict[str, Any]) -> Job:
    return Job("unbinding", {"N": int(N), "L": int(L), "n_structures": s["batch"],
                             "D_factors": Ds, "binarized": True},
               s["structures"] // s["batch"])


def _Ds(N: int, s: dict[str, Any]) -> list[int]:
    return s["Ds"] if N == s["Nb"] else [30]


def jobs(scale: str) -> list[Job]:
    s = SCALES[scale]
    Ns = sorted(set(s["Nc"]) | {s["Nb"]})
    return [_job(N, L, _Ds(N, s), s) for N in Ns for L in s["Ls"]]


def plot(ctx: Ctx) -> dict[str, Figure]:
    s = SCALES[ctx.scale]
    Ls = np.array(s["Ls"])
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    N = s["Nb"]
    for i, d in enumerate(s["Ds"]):
        pe = [ctx.load(_job(N, L, _Ds(N, s), s)) for L in Ls]
        ax_b.plot(Ls, [trials.stack(t, f"error_D{d}").mean() for t in pe],
                  label=f"$D={d}N$", **series(i))
        ax_b.plot(Ls, [theory.p_epsilon(theory.snr0(N, L, True), d * N) for L in Ls],
                  ls="--", lw=0.8, color=series(i)["color"])
    ax_b.set(xlabel="$L$", ylabel=r"$P_\epsilon$", title=f"$N={N}$")
    ax_b.legend(loc="upper left")

    for i, N in enumerate(s["Nc"]):
        inv = []
        for L in Ls:
            t = ctx.load(_job(N, L, _Ds(N, s), s))
            inv.append(1 / measures.snr(trials.stack(t, "signal_D30"),
                                        trials.stack(t, "noise_D30")))
        ax_c.plot(Ls, inv, label=f"$N={N}$", **series(i))
        ax_c.plot(Ls, [1 / theory.snr0(N, L, True) for L in Ls], ls="--", lw=0.8,
                  color=THEORY if i == 0 else series(i)["color"])
    ax_c.set(xlabel="$L$", ylabel=r"SNR$^{-1}$", title="$D=30N$")
    ax_c.legend(loc="upper left")
    fig.tight_layout()
    return {"fig1bc": fig}
