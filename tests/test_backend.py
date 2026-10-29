"""The optional torch backend against the numpy reference (skipped without torch)."""
from __future__ import annotations

import numpy as np
import pytest

from skam import backend, memory
from skam.vsa import Dictionary, ItemBank


@pytest.fixture
def torch_backend():
    pytest.importorskip("torch")
    backend.use("torch", "cpu" if not _has_gpu() else None)
    yield
    backend.use("numpy")


def _has_gpu() -> bool:
    import torch
    return bool(torch.cuda.is_available() or torch.backends.mps.is_available())


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        backend.use("jax")
    with pytest.raises(ValueError):
        backend.use("numpy", "cuda")


def _case():
    rng = np.random.default_rng(0)
    pats = rng.choice([-1.0, 1.0], size=(60, 400))
    S0 = np.where(rng.random((60, 400)) < 0.15, -pats, pats)
    return pats, S0


@pytest.mark.parametrize("rule", ["pseudo_inverse", "storkey"])
def test_parallel_dynamics_agree(torch_backend, rule):
    pats, S0 = _case()
    J = memory.learn(rule, pats)
    got = memory.run_parallel(J, S0, 20, targets=pats)
    backend.use("numpy")
    ref = memory.run_parallel(J, S0, 20, targets=pats)
    # float32 can only differ on fields within ~1e-6 of zero; none occur here
    np.testing.assert_array_equal(got.final, ref.final)
    np.testing.assert_allclose(got.overlap, ref.overlap, atol=1e-6)


def test_dictionary_scan_agrees(torch_backend):
    N, D = 64, 500
    bank = ItemBank(N, D, np.random.SeedSequence(3), block=128)
    rng = np.random.default_rng(1)
    true = rng.integers(0, D, size=40)
    a_hat = 0.4 * bank.get(true) + rng.normal(0, 1 / np.sqrt(N), size=(40, N))
    dic = Dictionary(bank, D, D)
    got = dic.explicit_overlaps(a_hat, true)
    backend.use("numpy")
    ref = dic.explicit_overlaps(a_hat, true)
    np.testing.assert_array_equal(got[1], ref[1])
    for g, r in zip((got[0], got[2], got[3]), (ref[0], ref[2], ref[3])):
        np.testing.assert_allclose(g, r, rtol=1e-4, atol=1e-5)
