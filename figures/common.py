"""Shared helpers for the figure modules: job builders, loading, style."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from skam import measures, trials
from skam.trials import Job, TrialResult

FloatArray = NDArray[np.float64]

# fixed categorical order (validated reference palette), marker as second channel
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
NEUTRAL = "#3d3d3a"
THEORY = "#3d3d3a"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

T_RETRIEVAL = 20          # "T = 20 parallel updates" in every caption
D_FACTOR = 30             # "The dictionary is fixed to D = 30N"


@dataclass(frozen=True)
class Ctx:
    """Which saved trials a figure is drawn from: ``Ctx("quick", seed=0)``."""
    scale: str
    seed: int
    results: Path = trials.DEFAULT_RESULTS
    backend: str = "numpy"

    def load(self, job: Job) -> list[TrialResult]:
        return trials.load(job, self.seed, self.results, self.backend)


def style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
        "lines.linewidth": 1.5, "lines.markersize": 4, "legend.frameon": False,
        "figure.dpi": 130, "savefig.bbox": "tight",
    })


def series(i: int) -> dict[str, Any]:
    return {"color": SERIES[i], "marker": MARKERS[i]}


def save(fig: Figure, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def trials_for(memories: int | None, P: int, fixed: int | None = None) -> int:
    """Networks needed to average over ``memories`` stored items, or ``fixed``
    networks when the scale sets a trial count directly (quick checks)."""
    if fixed is not None and memories is None:
        return fixed
    if memories is not None and fixed is None:
        return math.ceil(memories / P)
    raise ValueError("give exactly one of memories and fixed")


# ---------------------------------------------------------------- jobs
def structures_job(N: int, alpha: float, L: int, L0s: list[int], rule: str,
                   n_trials: int, decode: bool = True) -> Job:
    return Job("structures", {"N": int(N), "alpha": float(alpha), "L": int(L),
                              "L0s": [int(k) for k in L0s], "rule": rule,
                              "T": T_RETRIEVAL, "D_factor": D_FACTOR, "decode": decode},
               n_trials)


def random_job(N: int, alpha: float, rule: str, m0s: list[float], n_probe: int,
               n_trials: int, update: str = "parallel", beta: float | None = None,
               trajectory: bool = False, order_params: bool = False) -> Job:
    return Job("random_patterns", {
        "N": int(N), "alpha": float(alpha), "rule": rule, "update": update, "beta": beta,
        "T": T_RETRIEVAL, "m0s": [round(m, 6) for m in m0s], "n_probe": n_probe,
        "trajectory": trajectory, "order_params": order_params}, n_trials)


def grid(lo: float, hi: float, step: float) -> list[float]:
    n = int(round((hi - lo) / step))
    return [round(lo + k * step, 6) for k in range(n + 1)]


# ---------------------------------------------------------------- reductions
def decode_stats(tr: list[TrialResult], name: str) -> tuple[float, float]:
    """(P_eps, SNR) over all memories of all trials for read-out ``name``."""
    err = trials.stack(tr, f"error_{name}")
    sig = trials.stack(tr, f"signal_{name}")
    noise = trials.stack(tr, f"noise_{name}")
    return float(err.mean()), measures.snr(sig, noise)


def m_min(m0s: list[float], tr: list[TrialResult]) -> float:
    """m_min(alpha) from a random-pattern job: <m> with the reference fixed
    point reaches M_CONVERGED (skam.measures.convergence_threshold)."""
    mean_m = trials.stack(tr, "m_ref", axis=1).mean(axis=1)
    return measures.convergence_threshold(np.array(m0s), mean_m)
