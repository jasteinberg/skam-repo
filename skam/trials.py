"""Parallel, resumable execution of independent trials.

A trial is a pure function ``trial(point, seed) -> dict[str, ndarray]`` of a
parameter point (a JSON-serialisable dict) and a numpy SeedSequence. Trials
of all points run in a process pool; each writes one .npz.

Seeds. The SeedSequence of trial k of a point is derived from
(base_seed, hash(experiment, point), k), so a result does not depend on which
other points are run, on the number of workers, or on execution order.

Cache key. Results live in <results>/<experiment>/<key>/seed<s>_<backend>_trial<k>.npz
where key hashes the experiment name, the point and the source of the skam
package and of the experiment module. Changing any of them gives a new key, so
trials computed by different code are never mixed; rerunning skips trials on
disk. The backend (skam.backend) is part of the file name, so numpy and torch
results are never mixed either.

Points are checked against the experiment's KEYS: a missing or unknown key
raises instead of being ignored. Each trial file records the Python, numpy and
scipy versions (and torch, when used) that produced it; loading a point whose
trials come from different versions raises.

Sharding. ``shard=(i, n)`` runs only the trials whose hash of (key, k) is i
mod n. n machines running shards 0..n-1 of the same command, at the same
commit, together produce exactly the files of one unsharded run; merge them
by copying the results directories into one.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

Point = dict[str, Any]
TrialResult = dict[str, NDArray[Any]]
TrialFn = Callable[[Point, np.random.SeedSequence], TrialResult]

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_RESULTS = REPO_ROOT / "results"


@dataclass(frozen=True)
class Job:
    """``n_trials`` trials of ``experiment`` (a module in experiments/) at ``point``."""
    experiment: str
    point: Point
    n_trials: int


def _source_hash(experiment: str) -> str:
    h = hashlib.sha256()
    files = sorted(PACKAGE_DIR.glob("*.py")) + [REPO_ROOT / "experiments" / f"{experiment}.py"]
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def validate(experiment: str, point: Point) -> None:
    """Raise unless ``point`` has exactly the keys experiments.<experiment>.KEYS."""
    keys: frozenset[str] = importlib.import_module(f"experiments.{experiment}").KEYS
    missing, unknown = keys - set(point), set(point) - keys
    if missing or unknown:
        raise KeyError(f"{experiment} point: missing {sorted(missing)}, unknown {sorted(unknown)}")


def versions() -> str:
    """Library versions that determine a trial's numbers."""
    import scipy
    from . import backend
    v = f"python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}"
    if backend.name() == "torch":
        v += f", torch {backend.torch_module().__version__} on {backend.device()}"
    return v


def point_key(experiment: str, point: Point) -> str:
    blob = json.dumps({"experiment": experiment, "point": point,
                       "code": _source_hash(experiment)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def trial_seed(base_seed: int, experiment: str, point: Point, k: int) -> np.random.SeedSequence:
    digest = hashlib.sha256(json.dumps({"experiment": experiment, "point": point},
                                       sort_keys=True).encode()).digest()
    words = tuple(int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4))
    return np.random.SeedSequence(base_seed, spawn_key=words + (k,))


def point_dir(results: Path, job: Job) -> Path:
    return results / job.experiment / point_key(job.experiment, job.point)


def _trial_path(results: Path, job: Job, k: int, base_seed: int, backend_name: str) -> Path:
    return point_dir(results, job) / f"seed{base_seed}_{backend_name}_trial{k:05d}.npz"


def in_shard(key: str, k: int, shard: tuple[int, int]) -> bool:
    i, n = shard
    if not 0 <= i < n:
        raise ValueError(f"shard {i}/{n} invalid")
    return int(hashlib.sha256(f"{key}/{k}".encode()).hexdigest()[:8], 16) % n == i


def _init_worker(backend_name: str, device: str | None) -> None:
    from threadpoolctl import threadpool_limits
    from . import backend
    threadpool_limits(1)
    backend.use(backend_name, device)


def _execute(experiment: str, point: Point, seed: np.random.SeedSequence, path: str) -> float:
    validate(experiment, point)
    fn: TrialFn = importlib.import_module(f"experiments.{experiment}").trial
    t0 = time.perf_counter()
    out = fn(point, seed)
    for name, arr in out.items():
        if np.asarray(arr).dtype == object:
            raise TypeError(f"trial output {name!r} is not a numeric array")
    if "_versions" in out:
        raise KeyError("trial output may not be named _versions")
    out = {**out, "_versions": np.array(versions())}
    tmp = path + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)
    return time.perf_counter() - t0


def run(jobs: list[Job], base_seed: int, results: Path, n_workers: int,
        backend_name: str = "numpy", device: str | None = None,
        shard: tuple[int, int] = (0, 1)) -> None:
    """Run every missing trial of every job that falls in ``shard``."""
    todo: list[tuple[Job, int]] = []
    for job in jobs:
        validate(job.experiment, job.point)
        d = point_dir(results, job)
        d.mkdir(parents=True, exist_ok=True)
        (d / "point.json").write_text(json.dumps(
            {"experiment": job.experiment, "point": job.point,
             "code": _source_hash(job.experiment)}, indent=1, sort_keys=True))
        key = point_key(job.experiment, job.point)
        todo += [(job, k) for k in range(job.n_trials) if in_shard(key, k, shard)
                 and not _trial_path(results, job, k, base_seed, backend_name).exists()]
    total = sum(j.n_trials for j in jobs)
    print(f"{len(todo)} of {total} trials to run (shard {shard[0]}/{shard[1]}, "
          f"backend {backend_name}) on {n_workers} workers", flush=True)
    if not todo:
        return
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(backend_name, device)) as pool:
        futs = {pool.submit(_execute, job.experiment, job.point,
                            trial_seed(base_seed, job.experiment, job.point, k),
                            str(_trial_path(results, job, k, base_seed, backend_name))): (job, k)
                for job, k in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()                                  # re-raise worker errors
            if i % max(1, len(todo) // 20) == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} trials, {time.perf_counter() - t0:.0f}s", flush=True)


def load(job: Job, base_seed: int, results: Path, backend_name: str) -> list[TrialResult]:
    """All ``job.n_trials`` trials of a job; raises if any is missing or if
    the trials were produced by different library versions."""
    validate(job.experiment, job.point)
    out = []
    seen: set[str] = set()
    for k in range(job.n_trials):
        p = _trial_path(results, job, k, base_seed, backend_name)
        if not p.exists():
            raise FileNotFoundError(
                f"missing trial {k} of {job.experiment} {job.point} (seed {base_seed}, "
                f"backend {backend_name}): run the figure first, or merge all shards")
        with np.load(p) as z:
            out.append({name: z[name] for name in z.files if name != "_versions"})
            seen.add(str(z["_versions"]))
    if len(seen) > 1:
        raise ValueError(f"{job.experiment} {job.point}: trials from different library "
                         f"versions {sorted(seen)}; rerun them into one results directory")
    return out


def stack(trials: list[TrialResult], name: str, axis: int = 0) -> NDArray[Any]:
    """Concatenate one output across trials along ``axis``."""
    return np.concatenate([t[name] for t in trials], axis=axis)
