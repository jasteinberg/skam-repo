"""Each simulation on a small case with a known answer, plus runner guards.

These catch wiring errors (wrong relation unbound, wrong item scored, cue and
memory swapped) that would otherwise show up only as a plausible-looking curve.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments import ks_sequences, random_patterns, sequences, structures, unbinding
from skam import theory, trials
from skam.vsa import HRR, Binder, ItemBank, Structures


def _seed(k: int = 0) -> np.random.SeedSequence:
    return np.random.SeedSequence(1234, spawn_key=(k,))


def _signal_si32(N: int, L: int) -> float:
    """<a_1 . a_hat_1> from a binarized structure, SI Eq. 32: sqrt(2N / (pi L))."""
    return float(np.sqrt(2 * N / (np.pi * L)))


def _assert_mean_ratio(per_trial: list[float], expected: float) -> None:
    """Structures of one trial share their attribute vectors b_l (Methods), so
    the trial is the independent unit: compare the mean over trials with
    ``expected`` within 4 trial-level standard errors."""
    r = np.array(per_trial) / expected
    se = r.std(ddof=1) / np.sqrt(len(r))
    assert abs(r.mean() - 1) < 4 * se, (r.mean(), se)


def test_unbinding_matches_SI_eqs_32_34():
    N, L = 2048, 8
    outs = [unbinding.trial({"N": N, "L": L, "n_structures": 50, "D_factors": [30],
                             "binarized": True}, _seed(k)) for k in range(16)]
    _assert_mean_ratio([o["signal_D30"].mean() for o in outs], _signal_si32(N, L))
    sig = np.concatenate([o["signal_D30"] for o in outs])
    noise = np.concatenate([o["noise_D30"] for o in outs])
    # SI Eq. 34 is leading order; corrections are O(1/L) and O(L/N), so allow 15%
    assert abs(sig.mean() ** 2 / noise.mean() / theory.snr0(N, L, binarized=True) - 1) < 0.15
    assert not any(o["error_D30"].any() for o in outs)   # Eq. 21: P_eps ~ 1e-14 at SNR ~ 160


@pytest.mark.parametrize("rule", ["pseudo_inverse", "hebb", "storkey"])
def test_structures_long_cue_retrieves_and_reads_out_like_the_original(rule):
    N, L, L0 = 1024, 8, 6
    outs = [structures.trial({"N": N, "alpha": 0.02, "L": L, "L0s": [L0], "rule": rule,
                              "T": 20, "D_factor": 30, "decode": True}, _seed(k))
            for k in range(8)]
    _assert_mean_ratio([o[f"mcue_L0{L0}"].mean() for o in outs], theory.m0_of_cue(L0, L))
    _assert_mean_ratio([o["signal_original"].mean() for o in outs], _signal_si32(N, L))
    for out in outs:
        np.testing.assert_array_equal(out[f"m_L0{L0}"], 1.0)
        # retrieved state == stored pattern, so the read-out is identical
        np.testing.assert_array_equal(out[f"signal_L0{L0}"], out["signal_original"])
        assert not out["error_original"].any()


def test_sequences_decode_perfectly_and_shared_items_overlap_exactly():
    out = sequences.trial({"N": 2048, "P": 3, "L": 6, "n_shared": 2, "shared_start": 3,
                           "rule": "pseudo_inverse", "T": 20, "D_factor": 30}, _seed())
    assert out["correct"].all()
    np.testing.assert_array_equal(out["m"], 1.0)
    # positions 3-4 of sequence mu+1 hold the items of sequence mu
    for mu in range(2):
        np.testing.assert_array_equal(out["next_seq"][mu, 2:4], out["own"][mu, 2:4])
    unshared = out["next_seq"][:, [1, 4, 5]]
    assert abs(unshared.mean()) < 4 * unshared.std() / np.sqrt(unshared.size)


def test_ks_sequence_visits_patterns_in_order():
    out = ks_sequences.trial({"N": 1000, "L": 6, "shared": 4, "tau": 8, "lam": 2.5,
                              "T": 40, "rule": "pseudo_inverse"}, _seed())
    m = out["m"]
    assert m[0, 0] == 1.0
    winner = np.argmax(m, axis=1)
    first = [int(np.argmax(winner == p)) for p in range(3)]
    assert all(winner[f] == p for f, p in zip(first, range(3)))
    assert first[0] < first[1] < first[2]


def test_random_patterns_fixed_points_and_r():
    N, alpha = 500, 0.1
    out = random_patterns.trial({"N": N, "alpha": alpha, "rule": "pseudo_inverse",
                                 "update": "parallel", "beta": None, "T": 5, "m0s": [1.0],
                                 "n_probe": 50, "trajectory": False, "order_params": True},
                                _seed())
    np.testing.assert_array_equal(out["m"], 1.0)
    np.testing.assert_array_equal(out["m_ref"], 1.0)
    P = int(alpha * N)
    # E[r] at a stored pattern = (1/alpha) (P - 1) / N = (P - 1) / P
    assert abs(out["r"].mean() / ((P - 1) / P) - 1) < 0.1


class _PlainHRR(Binder):
    """HRR without the FFT prefix-sum override: exercises Binder.prefix_sums."""
    asymmetric = False

    def bind(self, a, b):
        return HRR(False).bind(a, b)

    def unbind(self, s, b):
        return HRR(False).unbind(s, b)


def test_generic_binder_encodes_like_hrr():
    N, P, L = 64, 5, 7
    bank = ItemBank(N, P * L, np.random.SeedSequence(4))
    attrs = np.random.default_rng(5).normal(size=(L, N))
    ids = np.arange(P * L).reshape(P, L)
    fast = Structures(bank, ids, attrs, HRR(False)).encode([0, 3, L])
    plain = Structures(bank, ids, attrs, _PlainHRR()).encode([0, 3, L])
    for n in (0, 3, L):
        np.testing.assert_allclose(plain[n], fast[n], atol=1e-10)


def test_point_keys_are_validated():
    good = {"N": 10, "L": 2, "shared": 1, "tau": 2, "lam": 2.0, "T": 3, "rule": "hebb"}
    trials.validate("ks_sequences", good)
    with pytest.raises(KeyError):
        trials.validate("ks_sequences", {**good, "D_factor": 30})
    with pytest.raises(KeyError):
        trials.validate("ks_sequences", {k: v for k, v in good.items() if k != "tau"})


def test_load_refuses_mixed_library_versions(tmp_path):
    job = trials.Job("ks_sequences", {"N": 40, "L": 3, "shared": 2, "tau": 2, "lam": 2.0,
                                      "T": 4, "rule": "hebb"}, 2)
    trials.run([job], 0, tmp_path, 1)
    trials.load(job, 0, tmp_path, "numpy")
    f = sorted(trials.point_dir(tmp_path, job).glob("*.npz"))[0]
    with np.load(f) as z:
        data = {k: z[k] for k in z.files}
    data["_versions"] = np.array("numpy 0.0")
    np.savez(f, **data)
    with pytest.raises(ValueError):
        trials.load(job, 0, tmp_path, "numpy")
