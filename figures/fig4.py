"""Fig. 4b: f(alpha) of Eq. 7, m* ~ f(alpha) m0 outside the basin.

Pseudo-inverse network of random patterns, N = 8000, T = 20 parallel updates,
50 trials. f(alpha) is the least-squares slope through the origin of m*
(mean overlap of the m < 1 mode) against m0, fitted on the small cue overlaps
m0 in FIT_M0 at which no state reaches the memory (p1 = 0); at least
MIN_FIT_POINTS such m0 are required. At the smallest alpha the basin edge
approaches FIT_M0 and m*(m0) turns superlinear, so f there depends on the range.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from skam import measures, trials
from skam.trials import Job

from .common import Ctx, Figure, NEUTRAL, random_job, series

FIT_M0 = [0.01, 0.02, 0.03, 0.04, 0.05]
MIN_FIT_POINTS = 3
SCALES: dict[str, dict[str, Any]] = {
    "paper": {"N": 8000, "trials": 50, "n_probe": 20,
              "alphas": [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25,
                         0.3, 0.35, 0.4, 0.45, 0.5]},
    "quick": {"N": 2000, "trials": 10, "n_probe": 20,
              "alphas": [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]},
}


def job(alpha: float, scale: str) -> Job:
    s = SCALES[scale]
    return random_job(s["N"], alpha, "pseudo_inverse", FIT_M0, s["n_probe"], s["trials"])


def jobs(scale: str) -> list[Job]:
    return [job(a, scale) for a in SCALES[scale]["alphas"]]


def f_alpha(ctx: Ctx, alpha: float) -> float:
    """Slope of m* vs m0 through the origin over FIT_M0."""
    N = SCALES[ctx.scale]["N"]
    m = trials.stack(ctx.load(job(alpha, ctx.scale)), "m", axis=1)
    shapes = [measures.overlap_shape(row, N) for row in m]
    out = np.array([sh.p1 == 0 for sh in shapes])
    if out.sum() < MIN_FIT_POINTS:
        raise ValueError(f"alpha={alpha}: only {out.sum()} of m0={FIT_M0} lie outside the basin")
    x = np.array(FIT_M0)[out]
    y = np.array([sh.m_star for sh in shapes])[out]
    return float(x @ y / (x @ x))


def plot(ctx: Ctx) -> dict[str, Figure]:
    alphas = SCALES[ctx.scale]["alphas"]
    f = [f_alpha(ctx, a) for a in alphas]
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.plot(alphas, f, **series(0))
    ax.axhline(1.0, ls="--", color=NEUTRAL, lw=0.8)
    ax.set(xlabel=r"$\alpha$", ylabel=r"$f(\alpha)$", title=f"$N={SCALES[ctx.scale]['N']}$")
    fig.tight_layout()
    return {"fig4b": fig}
