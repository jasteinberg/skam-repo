"""Run the simulations behind a figure, then draw it.

    python -m figures run  fig3 fig5 --scale quick --seed 0 --workers 8
    python -m figures plot fig3 fig5 --scale quick --seed 0
    python -m figures list --scale paper            # trial counts per figure

GPU (optional, needs PyTorch):  --backend torch [--device cuda|mps|cpu]
    Use few workers with a GPU (they share it), e.g. --workers 1 or 2.
Sharding across machines:       --shard i/n on machine i = 0..n-1, same command
    and commit; copy the results directories together before plotting.

``all`` selects every figure. ``--scale paper`` uses the published parameters;
``--scale quick`` the same panels at reduced N and trial counts. Results go to
results/ and figures to figures/out/<scale>/ unless --results / --out is given.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from types import ModuleType

import matplotlib
matplotlib.use("Agg")   # files only; notebooks keep their inline backend

from skam import trials  # noqa: E402

from .common import Ctx, save, style  # noqa: E402

FIGURES = ["fig1", "fig3", "fig4", "fig5", "fig6", "si"]
OUT_DEFAULT = Path(__file__).resolve().parent / "out"


def _modules(names: list[str]) -> list[ModuleType]:
    sel = FIGURES if names == ["all"] else names
    bad = [n for n in sel if n not in FIGURES]
    if bad:
        raise SystemExit(f"unknown figure(s) {bad}; choose from {FIGURES} or 'all'")
    return [importlib.import_module(f"figures.{n}") for n in sel]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "plot", "list"])
    ap.add_argument("figures", nargs="+")
    ap.add_argument("--scale", required=True, choices=["quick", "paper"])
    ap.add_argument("--seed", type=int, help="base seed (required for run and plot)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--results", type=Path, default=trials.DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--backend", choices=["numpy", "torch"], default="numpy",
                    help="numpy (reference, default) or torch (optional GPU)")
    ap.add_argument("--device", help="torch device; default cuda, else mps, else cpu")
    ap.add_argument("--shard", default="0/1", help="run only shard i of n, as i/n")
    a = ap.parse_args()
    try:
        si, sn = (int(x) for x in a.shard.split("/"))
    except ValueError:
        ap.error(f"--shard must look like i/n, got {a.shard!r}")
    mods = _modules(a.figures)
    if a.action != "list" and a.seed is None:
        ap.error("--seed is required")

    jobs: dict[str, trials.Job] = {}
    for m in mods:
        for j in m.jobs(a.scale):
            key = trials.point_key(j.experiment, j.point)
            if key not in jobs or jobs[key].n_trials < j.n_trials:
                jobs[key] = j                      # shared points: keep the largest trial count

    if a.action == "list":
        for m in mods:
            js = m.jobs(a.scale)
            print(f"{m.__name__}: {len(js)} points, {sum(j.n_trials for j in js)} trials")
        print(f"distinct points {len(jobs)}, trials {sum(j.n_trials for j in jobs.values())}")
    elif a.action == "run":
        trials.run(list(jobs.values()), a.seed, a.results, a.workers,
                   backend_name=a.backend, device=a.device, shard=(si, sn))
    else:
        style()
        ctx = Ctx(a.scale, a.seed, a.results, a.backend)
        for m in mods:
            for name, fig in m.plot(ctx).items():
                print("wrote", save(fig, a.out / a.scale, name))


if __name__ == "__main__":
    main()
