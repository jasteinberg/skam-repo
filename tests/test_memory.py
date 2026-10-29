"""Learning rules and dynamics against explicit formulas and exact cases."""
from __future__ import annotations

import numpy as np
import pytest

from skam import memory


def _patterns(P, N, seed):
    return np.random.default_rng(seed).choice([-1.0, 1.0], size=(P, N))


def test_pseudo_inverse_matches_pinv_formula():
    sig = _patterns(6, 23, 0)
    J = memory.pseudo_inverse(sig).dense()
    ref = sig.T @ np.linalg.pinv(sig @ sig.T / 23) @ sig / 23
    np.fill_diagonal(ref, 0)
    np.testing.assert_allclose(J, ref, atol=1e-12)
    np.testing.assert_allclose(J, J.T, atol=1e-12)


def test_pseudo_inverse_diagonal_averages_alpha():
    """sigma^T C^-1 sigma / N is the projector onto the pattern span: trace P."""
    sig = _patterns(30, 97, 1)
    assert abs(memory.pseudo_inverse(sig).diag.mean() - 30 / 97) < 1e-12


def test_pseudo_inverse_patterns_are_fixed_points():
    sig = _patterns(80, 400, 2)
    J = memory.pseudo_inverse(sig)
    out = memory.run_parallel(J, sig, 1).final
    np.testing.assert_array_equal(out, sig)


def test_low_rank_field_equals_dense_field():
    sig = _patterns(9, 41, 3)
    S = _patterns(5, 41, 4)
    for J in (memory.pseudo_inverse(sig), memory.hebb(sig)):
        np.testing.assert_allclose(J.field(S), S @ J.dense().T, atol=1e-12)


def test_hebb_matches_eq_26():
    sig = _patterns(4, 11, 5)
    ref = sig.T @ sig / 11
    np.fill_diagonal(ref, 0)
    np.testing.assert_allclose(memory.hebb(sig).dense(), ref, atol=1e-14)


def _storkey_loops(sig):
    """Eqs. 27-28 element by element."""
    P, N = sig.shape
    J = np.zeros((N, N))
    for mu in range(P):
        s = sig[mu]
        h = [sum(J[i, k] * s[k] for k in range(N)) for i in range(N)]
        new = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    new[i, j] = ((N + 1) / (N - 1) * J[i, j] + s[i] * s[j] / (N - 1)
                                 - (s[i] * h[j] + s[j] * h[i]) / (N - 1))
        J = new
    return J


def test_storkey_matches_elementwise_eq_27():
    sig = _patterns(3, 7, 6)
    np.testing.assert_allclose(memory.storkey(sig).dense(), _storkey_loops(sig), atol=1e-12)


def _parallel_loop(J, s, T):
    s = s.copy()
    for _ in range(T):
        h = J @ s
        s = np.array([1.0 if h[i] > 0 else -1.0 if h[i] < 0 else s[i] for i in range(len(s))])
    return s


def test_parallel_dynamics_matches_loop_including_ties():
    # Hebb with an even number of patterns produces exact zero fields
    sig = _patterns(2, 9, 7)
    J = memory.hebb(sig)
    S0 = _patterns(6, 9, 8)
    out = memory.run_parallel(J, S0, 5).final
    for b in range(6):
        np.testing.assert_array_equal(out[b], _parallel_loop(J.dense(), S0[b], 5))


def test_metropolis_at_beta_zero_flips_every_spin_once_per_sweep():
    sig = _patterns(5, 30, 9)
    S0 = _patterns(4, 30, 10)
    J = memory.pseudo_inverse(sig)
    out = memory.run_serial(J, S0, 1, beta=0.0, rng=np.random.default_rng(0)).final
    np.testing.assert_array_equal(out, -S0)


def test_zero_noise_serial_reaches_a_fixed_point():
    sig = _patterns(20, 200, 11)
    J = memory.pseudo_inverse(sig)
    S0 = _patterns(10, 200, 12)
    out = memory.run_serial(J, S0, 50, beta=None, rng=np.random.default_rng(1)).final
    h = J.field(out)
    assert np.all(h * out >= 0)


def test_trajectory_records_overlap():
    sig = _patterns(10, 100, 13)
    J = memory.pseudo_inverse(sig)
    tr = memory.run_parallel(J, sig, 3, targets=sig)
    assert tr.overlap is not None and tr.overlap.shape == (4, 10)
    np.testing.assert_array_equal(tr.overlap, 1.0)


def test_unknown_rule_and_update_raise():
    with pytest.raises(ValueError):
        memory.learn("oja", _patterns(2, 5, 0))
    with pytest.raises(ValueError):
        memory.retrieve(memory.hebb(_patterns(2, 5, 0)), _patterns(1, 5, 1), 1, "async")
