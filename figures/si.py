"""Supplementary Figs. 1-8.

SI 1a  m_min(alpha), pseudo-inverse, random patterns; N = 1000, 50 trials. alpha < 1 only:
       at alpha = 1 the pseudo-inverse J vanishes (projector onto all of R^N, minus its diagonal).
SI 1b  l_c(alpha) for structures, L in {20, 30, 40}, against Eq. 31 with m_min from 1a.
       l_c = smallest L0/L at which <m> reaches 0.99 (same criterion as m_min).
SI 2   p1, m*, sigma_m^2 vs m0, N in {500..16000}, alpha in {0.1, 0.2}; 50 trials.
SI 3a  p(m) for structures, alpha = 0.1, L0 = 2, L in {8, 16, 24}, N in {1000, 8000}; 100 trials.
SI 3b  p(m) for random patterns at the m0 of Eq. 30 for the same L0, L.
SI 4   <m(t)> for parallel and serial (beta = 5, 10, inf) updates; alpha = 0.1, N = 1000.
SI 5   <m>, <r>, <m^0> after T = 20 vs m0 for the same dynamics.
SI 6   <m> vs m0 for Hebb, Storkey, pseudo-inverse; alpha in {0.05, 0.1}, N = 1000.
SI 7   m_min vs alpha for the three rules.
SI 8a  P_eps vs L, three rules, N = 1000, alpha = 0.1, L0 = 2, 20,000 memories.
SI 8b  (SNR/SNR0)^-1 vs L/L0, three rules, N = 8000, L0 = 2.

m_min is the smallest m0 at which <m> with the memory's fixed point reaches 0.99
("the minimal overlap ... required for convergence to the correct fixed point on
average", main text); skam.measures.convergence_threshold. This reproduces the
published SI Fig. 1a; the midpoint p1 = 1/2 of Eq. 33 lies 0.03-0.07 lower at N = 1000.
sigma_m^2 is the variance of the m < 1 mode, as defined in Methods.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from skam import measures, theory, trials
from skam.trials import Job

from .common import (BLUES, Figure, NEUTRAL, SERIES, THEORY, Ctx, decode_stats, grid, m_min,
                     random_job, series, structures_job, trials_for)

RULES = ["hebb", "storkey", "pseudo_inverse"]
RULE_LABEL = {"hebb": "Hebb", "storkey": "Storkey", "pseudo_inverse": "Pseudo-inverse"}

SCALES: dict[str, dict[str, Any]] = {
    "paper": {
        "N": 1000, "trials": 50, "n_probe": 20,
        "alphas_1a": grid(0.05, 0.975, 0.025), "alphas_1b": grid(0.05, 0.4, 0.025),
        "Ls_1b": [20, 30, 40],
        "Ns_2": [500, 1000, 2000, 4000, 8000, 16000], "trials_3": 100, "Ns_3": [1000, 8000],
        "trials_45": 50, "alphas_7": grid(0.01, 0.1, 0.005),
        "memories_8": 20000, "trials_8": None, "N_8b": 8000, "Ls_8a": list(range(4, 31, 2)),
        "Ls_8b": list(range(4, 41, 2)),
    },
    "quick": {
        "N": 1000, "trials": 10, "n_probe": 20,
        "alphas_1a": grid(0.05, 0.95, 0.05), "alphas_1b": grid(0.05, 0.4, 0.05),
        "Ls_1b": [20, 30],
        "Ns_2": [500, 1000, 2000], "trials_3": 10, "Ns_3": [1000, 2000],
        "trials_45": 10, "alphas_7": grid(0.01, 0.1, 0.01),
        "memories_8": None, "trials_8": 10, "N_8b": 2000, "Ls_8a": list(range(4, 31, 2)),
        "Ls_8b": list(range(4, 41, 4)),
    },
}
M0_FULL = grid(0.0, 1.0, 0.02)
M0_FINE = grid(0.0, 0.4, 0.01) + grid(0.45, 1.0, 0.05)
DYNAMICS = [("parallel", None), ("serial", 5.0), ("serial", 10.0), ("serial", None)]


def _dyn_label(update: str, beta: float | None) -> str:
    if update == "parallel":
        return "Parallel"
    return r"$\beta=\infty$" if beta is None else f"$\\beta={beta:g}$"


# ---------------------------------------------------------------- job builders
def _mmin_job(s: dict[str, Any], alpha: float, rule: str, m0s: list[float] = M0_FINE) -> Job:
    return random_job(s["N"], alpha, rule, m0s, s["n_probe"], s["trials"])


def _lc_job(s: dict[str, Any], alpha: float, L: int) -> Job:
    return structures_job(s["N"], alpha, L, list(range(1, L)), "pseudo_inverse",
                          s["trials"], decode=False)


def _si2_job(s: dict[str, Any], N: int, alpha: float) -> Job:
    return random_job(N, alpha, "pseudo_inverse", M0_FINE, s["n_probe"], s["trials"])


def _si3a_job(s: dict[str, Any], N: int, L: int) -> Job:
    return structures_job(N, 0.1, L, [1, 2, 3], "pseudo_inverse", s["trials_3"])


def _si3b_job(s: dict[str, Any], N: int) -> Job:
    m0s = [theory.m0_of_cue(2, L) for L in (8, 16, 24)]
    return random_job(N, 0.1, "pseudo_inverse", m0s, int(round(0.1 * N)), s["trials_3"])


def _dyn_job(s: dict[str, Any], update: str, beta: float | None) -> Job:
    return random_job(s["N"], 0.1, "pseudo_inverse", M0_FULL, int(round(0.1 * s["N"])),
                      s["trials_45"], update=update, beta=beta, trajectory=True,
                      order_params=True)


def _rule_job(s: dict[str, Any], alpha: float, rule: str) -> Job:
    return random_job(s["N"], alpha, rule, M0_FULL, s["n_probe"], s["trials"])


def _si8_job(s: dict[str, Any], N: int, L: int, rule: str, L0s: list[int]) -> Job:
    P = int(round(0.1 * N))
    return structures_job(N, 0.1, L, L0s, rule, trials_for(s["memories_8"], P, s["trials_8"]))


def jobs(scale: str) -> list[Job]:
    s = SCALES[scale]
    out = [_mmin_job(s, a, "pseudo_inverse") for a in s["alphas_1a"]]
    out += [_lc_job(s, a, L) for a in s["alphas_1b"] for L in s["Ls_1b"]]
    out += [_si2_job(s, N, a) for N in s["Ns_2"] for a in (0.1, 0.2)]
    out += [_si3a_job(s, N, L) for N in s["Ns_3"] for L in (8, 16, 24)]
    out += [_si3b_job(s, N) for N in s["Ns_3"]]
    out += [_dyn_job(s, u, b) for u, b in DYNAMICS]
    out += [_rule_job(s, a, r) for a in (0.05, 0.1) for r in RULES]
    out += [_mmin_job(s, a, r) for a in s["alphas_7"] for r in RULES]
    out += [_si8_job(s, s["N"], L, r, [1, 2, 3]) for L in s["Ls_8a"] for r in RULES]
    out += [_si8_job(s, s["N_8b"], L, r, [2]) for L in s["Ls_8b"] for r in RULES]
    return out


# ---------------------------------------------------------------- figures
def _m_min_of(ctx: Ctx, s: dict[str, Any], alpha: float, rule: str) -> float:
    return m_min(M0_FINE, ctx.load(_mmin_job(s, alpha, rule)))


def si1(ctx: Ctx) -> Figure:
    s = SCALES[ctx.scale]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.8))
    al = s["alphas_1a"]
    mm = [_m_min_of(ctx, s, x, "pseudo_inverse") for x in al]
    a.plot(al, mm, **series(0))
    a.set(xlabel=r"$\alpha$", ylabel=r"$m_{min}$", title="(a) random patterns")
    for i, L in enumerate(s["Ls_1b"]):
        lc = []
        for x in s["alphas_1b"]:
            t = ctx.load(_lc_job(s, x, L))
            mean_m = np.array([trials.stack(t, f"m_L0{k}").mean() for k in range(1, L)])
            lc.append(measures.convergence_threshold(np.arange(1, L) / L, mean_m))
        b.plot(s["alphas_1b"], lc, label=f"$L={L}$", **series(i))
    pred = [theory.lc_of_mmin(float(np.interp(x, al, mm))) for x in s["alphas_1b"]]
    b.plot(s["alphas_1b"], pred, ls="--", color=THEORY, label=r"Eq. 31 from $m_{min}$")
    b.set(xlabel=r"$\alpha$", ylabel=r"$l_c$", title="(b) structures")
    b.legend()
    fig.tight_layout()
    return fig


def si2(ctx: Ctx) -> Figure:
    s = SCALES[ctx.scale]
    fig, ax = plt.subplots(2, 3, figsize=(7.4, 4.8))
    m0 = np.array(M0_FINE)
    for r, alpha in enumerate((0.1, 0.2)):
        for i, N in enumerate(s["Ns_2"]):
            m = trials.stack(ctx.load(_si2_job(s, N, alpha)), "m", axis=1)
            sh = [measures.overlap_shape(row, N) for row in m]
            kw = dict(label=f"$N={N}$", **series(i))
            ax[r, 0].plot(m0, [x.p1 for x in sh], **kw)
            lo = m0 <= {0.1: 0.2, 0.2: 0.34}[alpha]
            ax[r, 1].plot(m0[lo], np.array([x.m_star for x in sh])[lo], **kw)
            ax[r, 2].plot(m0[lo], np.array([x.var_m for x in sh])[lo], **kw)
        ax[r, 1].plot([0, m0[lo].max()], [0, m0[lo].max()], ls="--", color=NEUTRAL, lw=0.8)
        ax[r, 0].set(xlabel="$m_0$", ylabel="$p_1$")
        ax[r, 1].set(xlabel="$m_0$", ylabel="$m^*$", title=f"$\\alpha={alpha}$")
        ax[r, 2].set(xlabel="$m_0$", ylabel=r"$\sigma_m^2$")
    ax[0, 0].legend(fontsize=6.5)
    fig.tight_layout()
    return fig


def si3(ctx: Ctx) -> Figure:
    s = SCALES[ctx.scale]
    Ns = s["Ns_3"]
    fig, ax = plt.subplots(2 * len(Ns), 3, figsize=(7.0, 2.0 * 2 * len(Ns)))
    bins = np.linspace(0.0, 1.0, 101)
    for n_i, N in enumerate(Ns):
        rb = trials.stack(ctx.load(_si3b_job(s, N)), "m", axis=1)
        for c, L in enumerate((8, 16, 24)):
            ma = trials.stack(ctx.load(_si3a_job(s, N, L)), "m_L02")
            for row, data, title in ((n_i, ma, f"$N={N}$, $L={L}$"),
                                     (len(Ns) + n_i, rb[c],
                                      f"$N={N}$, $m_0={theory.m0_of_cue(2, L):.2f}$")):
                w = np.full(data.size, 1.0 / data.size)
                ax[row, c].hist(data, bins=bins, weights=w, color=SERIES[0])
                ax[row, c].set(xlabel="$m$", ylabel="$p(m)$", title=title)
    fig.tight_layout()
    return fig


def si4(ctx: Ctx) -> Figure:
    s = SCALES[ctx.scale]
    shown = [m for m in M0_FULL if 0 < m <= 0.5 and round(m * 50) % 2 == 1]  # 0.02, 0.06, ...
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0))
    for ax, (u, b) in zip(axes.ravel(), DYNAMICS):
        mt = trials.stack(ctx.load(_dyn_job(s, u, b)), "m_t", axis=2)       # (T+1, n_m0, ...)
        for m0 in shown:
            k = M0_FULL.index(m0)
            ax.plot(np.arange(mt.shape[0]), mt[:, k].mean(axis=1), marker=".", ms=3,
                    color=cmap(m0 / 0.5))
        ax.set(xlabel="$t$", ylabel=r"$\langle m(t)\rangle$", title=_dyn_label(u, b),
               ylim=(0, 1.02))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 0.5))
    fig.colorbar(sm, ax=axes, label="$m_0$", shrink=0.8)
    return fig


def si5(ctx: Ctx) -> Figure:
    s = SCALES[ctx.scale]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
    m0 = np.array(M0_FULL)
    for i, (u, b) in enumerate(DYNAMICS):
        t = ctx.load(_dyn_job(s, u, b))
        kw = dict(label=_dyn_label(u, b), **series(i))
        for a, name in zip(ax, ("m", "r", "m_init")):
            a.plot(m0, trials.stack(t, name, axis=1).mean(axis=1), **kw)
    for a, lab in zip(ax, (r"$\langle m\rangle$", r"$\langle r\rangle$", r"$\langle m^0\rangle$")):
        a.set(xlabel="$m_0$", ylabel=lab)
    for a in (ax[0], ax[2]):
        a.plot([0, 1], [0, 1], ls="--", color=NEUTRAL, lw=0.8)
    ax[0].legend()
    fig.tight_layout()
    return fig


def si6(ctx: Ctx) -> Figure:
    s = SCALES[ctx.scale]
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.8))
    for a, alpha in zip(ax, (0.05, 0.1)):
        for i, r in enumerate(RULES):
            m = trials.stack(ctx.load(_rule_job(s, alpha, r)), "m", axis=1).mean(axis=1)
            a.plot(M0_FULL, m, label=RULE_LABEL[r], **series(i))
        a.plot([0, 1], [0, 1], ls="--", color=NEUTRAL, lw=0.8)
        a.set(xlabel="$m_0$", ylabel=r"$\langle m\rangle$", title=f"$\\alpha={alpha}$")
    ax[0].legend()
    fig.tight_layout()
    return fig


def si7(ctx: Ctx) -> Figure:
    s = SCALES[ctx.scale]
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    for i, r in enumerate(RULES):
        ax.plot(s["alphas_7"], [_m_min_of(ctx, s, a, r) for a in s["alphas_7"]],
                label=RULE_LABEL[r], **series(i))
    ax.set(xlabel=r"$\alpha$", ylabel=r"$m_{min}$")
    ax.legend()
    fig.tight_layout()
    return fig


def si8(ctx: Ctx) -> Figure:
    s = SCALES[ctx.scale]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.8))
    N = s["N"]
    La = np.array(s["Ls_8a"])
    for i, r in enumerate(RULES):
        pe = [decode_stats(ctx.load(_si8_job(s, N, L, r, [1, 2, 3])), "L02")[0] for L in La]
        a.plot(La, pe, label=RULE_LABEL[r], **series(i))
    pe0 = [decode_stats(ctx.load(_si8_job(s, N, L, "pseudo_inverse", [1, 2, 3])), "original")[0]
           for L in La]
    a.plot(La, pe0, color=NEUTRAL, marker=".", label="Original")
    a.set(xlabel="$L$", ylabel=r"$P_\epsilon$", title=f"(a) $N={N}$")
    a.legend()
    Nb, Lb = s["N_8b"], np.array(s["Ls_8b"])
    for i, r in enumerate(RULES):
        snr = np.array([decode_stats(ctx.load(_si8_job(s, Nb, L, r, [2])), "L02")[1] for L in Lb])
        b.plot(Lb / 2, (Nb / Lb) / snr, label=RULE_LABEL[r], **series(i))
    b.set(xlabel="$L/L_0$", ylabel=r"$(\mathrm{SNR}/\mathrm{SNR}_0)^{-1}$", title=f"(b) $N={Nb}$")
    b.legend()
    fig.tight_layout()
    return fig


def plot(ctx: Ctx) -> dict[str, Figure]:
    return {f"si_fig{i}": f(ctx) for i, f in enumerate((si1, si2, si3, si4, si5, si6, si7, si8), 1)}
