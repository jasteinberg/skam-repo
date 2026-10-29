"""Fig. 5: SNR^-1 vs L^2 / (N L0), against Eq. 9 with c = 0.65.

N in {8000, 12000}, alpha in {0.1, 0.2}, L0 in {1, 2, 3}, 10,000 memories,
T = 20 parallel updates. The theory line uses f(alpha) measured in Fig. 4b.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from skam import theory
from skam.trials import Job

from . import fig3, fig4
from .common import Ctx, Figure, THEORY, decode_stats, series

SCALES: dict[str, dict[str, Any]] = {
    "paper": {"Ns": [8000, 12000], "alphas": [0.1, 0.2], "Ls": list(range(4, 41, 2)),
              "memories": 10000, "trials": None},
    "quick": {"Ns": [2000, 3000], "alphas": [0.1, 0.2], "Ls": list(range(4, 41, 4)),
              "memories": None, "trials": 10},
}
X_MAX = 0.2


def jobs(scale: str) -> list[Job]:
    s = SCALES[scale]
    own = [fig3.job(N, a, L, s) for N in s["Ns"] for a in s["alphas"] for L in s["Ls"]]
    missing = [a for a in s["alphas"] if a not in fig4.SCALES[scale]["alphas"]]
    if missing:
        raise ValueError(f"Fig. 4b grid lacks alphas {missing} needed for the theory line")
    return own + [fig4.job(a, scale) for a in s["alphas"]]


def plot(ctx: Ctx) -> dict[str, Figure]:
    s = SCALES[ctx.scale]
    Ls = np.array(s["Ls"])
    fig, axes = plt.subplots(1, len(s["alphas"]), figsize=(3.2 * len(s["alphas"]), 3.0))
    for ax, a in zip(axes, s["alphas"]):
        for n_i, N in enumerate(s["Ns"]):
            tr = [ctx.load(fig3.job(N, a, L, s)) for L in Ls]
            for i, L0 in enumerate(fig3.L0S):
                snr = np.array([decode_stats(t, f"L0{L0}")[1] for t in tr])
                x = Ls ** 2 / (N * L0)
                keep = x <= X_MAX
                ax.plot(x[keep], 1 / snr[keep], ls="none", color=series(i)["color"],
                        marker="x" if n_i == 0 else "s", mfc="none",
                        label=f"$N={N}$, $L_0={L0}$")
        f = fig4.f_alpha(ctx, a)
        xx = np.linspace(0, X_MAX, 50)
        # Eq. 9 is SNR^-1 = k L^2/(N L0) with k = SNR^-1 at N = L = L0 = 1
        k = 1 / theory.snr_out_of_basin(N=1, L=1, L0=1, f_alpha=f)
        ax.plot(xx, k * xx, color=THEORY, label=f"Eq. 9, $f={f:.2f}$")
        ax.set(xlabel=r"$L^2/NL_0$", ylabel=r"SNR$^{-1}$", title=f"$\\alpha={a}$")
    axes[0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    return {"fig5": fig}
