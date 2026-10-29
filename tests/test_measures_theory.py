"""Order parameters and analytic formulas against hand cases and closed forms."""
from __future__ import annotations

import numpy as np
import pytest

from skam import ks, measures, memory, theory


def test_flip_to_overlap_is_exact():
    pats = np.random.default_rng(0).choice([-1.0, 1.0], size=(7, 101))
    for m0 in (0.0, 0.3, 1.0):
        S = measures.flip_to_overlap(pats, m0, np.random.default_rng(1))
        k = round(101 * (1 - m0) / 2)
        np.testing.assert_allclose(measures.overlap(S, pats), 1 - 2 * k / 101)


def test_order_r_hand_case():
    pats = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1]], float)
    S = pats[:1].copy()
    # m_nu = 0 for nu = 1, 2 (orthogonal rows)
    assert measures.order_r(S, pats, np.array([0]))[0] == 0.0
    S = np.array([[1, 1, 1, -1]], float)    # m = (0.5, 0.5, 0.5)
    assert measures.order_r(S, pats, np.array([0]))[0] == pytest.approx((0.25 + 0.25) / 0.75)


def test_overlap_shape_and_crossing():
    sh = measures.overlap_shape(np.array([1.0, 1.0, 0.2, 0.4]), N=10)
    assert sh.p1 == 0.5 and sh.m_star == pytest.approx(0.3) and sh.var_m == pytest.approx(0.01)
    assert measures.first_crossing(np.array([0, 1, 2.0]), np.array([0, 0.25, 1.0]), 0.5) \
        == pytest.approx(1 + 0.25 / 0.75)
    with pytest.raises(ValueError):
        measures.first_crossing(np.array([0, 1.0]), np.array([0, 0.1]), 0.5)


def test_snr_is_ratio_of_means():
    assert measures.snr(np.array([1.0, 3.0]), np.array([1.0, 3.0])) == pytest.approx(4 / 2)


def test_p_epsilon_zero_snr_is_chance():
    """At SNR = 0 the true item is one of D+1 exchangeable Gaussians."""
    for D in (1, 9, 1000):
        assert theory.p_epsilon(0.0, D) == pytest.approx(D / (D + 1), abs=1e-7)


def test_p_epsilon_matches_monte_carlo_of_its_definition():
    rng = np.random.default_rng(0)
    snr, D, M = 12.0, 200, 40000
    signal = np.sqrt(snr) + rng.normal(size=M)
    mx = rng.normal(size=(M, D)).max(axis=1)
    p = np.mean(mx >= signal)
    assert abs(theory.p_epsilon(snr, D) - p) < 4 * np.sqrt(p * (1 - p) / M)


def test_lc_inverts_m0_of_cue():
    for x in (0.05, 0.3, 0.8):
        assert theory.lc_of_mmin(theory.m0_of_cue(x, 1.0)) == pytest.approx(x)


def test_ks_dwell_time_eq_89():
    """A single long sequence advances one pattern per t0 = (tau/2)(1 + 1/lambda)."""
    N, L, tau, lam = 2000, 12, 8, 2.5
    pats = np.random.default_rng(3).choice([-1.0, 1.0], size=(L, N))
    J1, J2 = ks.build(pats, np.arange(L)[None], lam, "hebb")
    hist = ks.run(J1, J2, pats[:1], 80, tau)
    winner = np.argmax(hist[:, 0] @ pats.T, axis=1)
    entries = [int(np.argmax(winner == l)) for l in range(3, 9)]
    period = np.diff(entries).mean()
    assert abs(period - theory.ks_period(tau, lam)) <= 1.0


def test_ks_transition_couplings_hand_case():
    pats = np.array([[1, -1, 1], [1, 1, -1]], float)
    J2 = ks.transition_couplings(pats, np.array([[0, 1]]), lam=3.0)
    ref = 3.0 / 3 * np.outer(pats[1], pats[0])
    np.fill_diagonal(ref, 0)
    np.testing.assert_allclose(J2, ref)


def test_learn_dispatch():
    pats = np.random.default_rng(0).choice([-1.0, 1.0], size=(3, 20))
    for rule in memory.RULES:
        J = memory.learn(rule, pats).dense()
        assert np.all(np.diag(J) == 0)
