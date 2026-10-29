"""Fig. 3: decoding error and SNR of a non-cued relation after retrieval.

Columns: (N, alpha) in {1000, 8000} x {0.1, 0.2}; cue lengths L0 = 1, 2, 3 and
the original (stored, unretrieved) structure. T = 20 parallel updates, 10,000
memories, D = 30N.
(a) P_eps vs L. (b) SNR^-1 vs L. (c) (SNR/SNR0)^-1 vs L/L0 with SNR0 = N/L.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from skam.trials import Job

from .common import Ctx, Figure, NEUTRAL, decode_stats, series, structures_job, trials_for

L0S = [1, 2, 3]
SCALES: dict[str, dict[str, Any]] = {
    "paper": {"Ns": [1000, 8000], "alphas": [0.1, 0.2], "Ls": list(range(4, 31, 2)),
              "memories": 10000, "trials": None},
    "quick": {"Ns": [1000, 2000], "alphas": [0.1, 0.2], "Ls": list(range(4, 31, 2)),
              "memories": None, "trials": 10},
}


def job(N: int, alpha: float, L: int, s: dict[str, Any]) -> Job:
    P = int(round(alpha * N))
    return structures_job(N, alpha, L, L0S, "pseudo_inverse",
                          trials_for(s["memories"], P, s["trials"]))


def jobs(scale: str) -> list[Job]:
    s = SCALES[scale]
    return [job(N, a, L, s) for N in s["Ns"] for a in s["alphas"] for L in s["Ls"]]


def plot(ctx: Ctx) -> dict[str, Figure]:
    s = SCALES[ctx.scale]
    Ls = np.array(s["Ls"])
    cols = [(N, a) for N in s["Ns"] for a in s["alphas"]]
    fig, axes = plt.subplots(3, len(cols), figsize=(2.3 * len(cols), 6.6))
    for c, (N, a) in enumerate(cols):
        tr = [ctx.load(job(N, a, L, s)) for L in Ls]
        for i, L0 in enumerate(L0S + ["original"]):
            name = f"L0{L0}" if L0 != "original" else "original"
            pe, snr = np.array([decode_stats(t, name) for t in tr]).T
            label = f"$L_0={L0}$" if L0 != "original" else "Original"
            kw = series(i) if L0 != "original" else {"color": NEUTRAL, "marker": "."}
            axes[0, c].plot(Ls, pe, label=label, **kw)
            axes[1, c].plot(Ls, 1 / snr, label=label, **kw)
            if L0 != "original":
                axes[2, c].plot(Ls / L0, (N / Ls) / snr, label=label, **kw)
        axes[0, c].set_title(f"$N={N}$, $\\alpha={a}$")
        axes[0, c].set(xlabel="$L$", ylabel=r"$P_\epsilon$")
        axes[1, c].set(xlabel="$L$", ylabel=r"SNR$^{-1}$")
        axes[2, c].set(xlabel="$L/L_0$", ylabel=r"$(\mathrm{SNR}/\mathrm{SNR}_0)^{-1}$")
    for r in range(3):
        axes[r, 0].legend(loc="upper left")
    fig.tight_layout()
    return {"fig3": fig}
