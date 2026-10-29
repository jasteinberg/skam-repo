"""Seeding, caching and repository hygiene."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from skam import trials

REPO = Path(__file__).resolve().parents[1]


def _job(n=3):
    return trials.Job("random_patterns", {
        "N": 60, "alpha": 0.1, "rule": "pseudo_inverse", "update": "parallel",
        "beta": None, "T": 5, "m0s": [0.2, 0.8], "n_probe": 3,
        "trajectory": True, "order_params": True}, n)


def test_results_independent_of_worker_count(tmp_path):
    job = _job()
    trials.run([job], base_seed=0, results=tmp_path / "a", n_workers=1)
    trials.run([job], base_seed=0, results=tmp_path / "b", n_workers=3)
    a = trials.load(job, 0, tmp_path / "a", "numpy")
    b = trials.load(job, 0, tmp_path / "b", "numpy")
    for ta, tb in zip(a, b):
        for k in ta:
            np.testing.assert_array_equal(ta[k], tb[k])


def test_seeds_differ_across_trials_points_and_base_seed():
    p = _job().point
    s = [trials.trial_seed(0, "x", p, 0), trials.trial_seed(0, "x", p, 1),
         trials.trial_seed(1, "x", p, 0), trials.trial_seed(0, "x", {**p, "N": 61}, 0)]
    draws = [np.random.default_rng(x).random() for x in s]
    assert len(set(draws)) == 4


def test_load_raises_on_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        trials.load(_job(), 0, tmp_path, "numpy")


def test_shards_partition_the_trials_and_merge_to_the_full_run(tmp_path):
    import shutil
    job = _job(n=7)
    trials.run([job], 0, tmp_path / "full", 1)
    for i in range(3):
        trials.run([job], 0, tmp_path / f"s{i}", 1, shard=(i, 3))
        shutil.copytree(tmp_path / f"s{i}", tmp_path / "merged", dirs_exist_ok=True)
    counts = [len(list((tmp_path / f"s{i}").rglob("*.npz"))) for i in range(3)]
    assert sum(counts) == 7
    for ta, tb in zip(trials.load(job, 0, tmp_path / "full", "numpy"),
                      trials.load(job, 0, tmp_path / "merged", "numpy")):
        for k in ta:
            np.testing.assert_array_equal(ta[k], tb[k])


def test_no_absolute_paths_in_tracked_files():
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                           text=True, check=True).stdout.split()
    # assembled from pieces so this file does not match itself
    roots = ["/" + "Users/", "/" + "home/", "/" + "private/var", "~" + "/"]
    pat = re.compile("|".join(re.escape(r) for r in roots))
    bad = []
    for f in files:
        p = REPO / f
        if p.suffix in {".pdf", ".png", ".npz"} or not p.is_file():
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if pat.search(line):
                bad.append(f"{f}:{i}")
    assert not bad, bad
