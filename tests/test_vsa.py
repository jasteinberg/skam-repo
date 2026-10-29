"""Binding, Dictionary and structure encoding, against closed forms and loops."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from skam import theory
from skam.vsa import HRR, Dictionary, ItemBank, Sequences, Structures, binarize


def _loop_conv(a, b):
    N = len(a)
    return np.array([sum(a[j] * b[(k - j) % N] for j in range(N)) for k in range(N)])


def _loop_corr(s, b):
    """Plate's correlation a_hat_k = sum_j b_j s_{j+k}."""
    N = len(s)
    return np.array([sum(b[j] * s[(j + k) % N] for j in range(N)) for k in range(N)])


def test_symmetric_bind_unbind_match_index_sums():
    rng = np.random.default_rng(0)
    a, b, s = rng.normal(size=(3, 7))
    h = HRR(asymmetric=False)
    np.testing.assert_allclose(h.bind(a, b), _loop_conv(a, b), atol=1e-12)
    np.testing.assert_allclose(h.unbind(s, b), _loop_corr(s, b), atol=1e-12)
    np.testing.assert_allclose(h.bind(a, b), h.bind(b, a), atol=1e-12)     # commutative


def test_asymmetric_binding_swaps_operations():
    rng = np.random.default_rng(1)
    a, b, s = rng.normal(size=(3, 7))
    h = HRR(asymmetric=True)
    np.testing.assert_allclose(h.bind(a, b), _loop_corr(a, b), atol=1e-12)
    np.testing.assert_allclose(h.unbind(s, b), _loop_conv(s, b), atol=1e-12)
    assert not np.allclose(h.bind(a, b), h.bind(b, a))


@pytest.mark.parametrize("asym", [False, True])
def test_unbinding_moments_match_SI_eqs_19_20(asym):
    """<a_1 . a_hat_1> = 1 and Var = (L+5)/N + 2/N^2 (SI Eqs. 19-20).
    The variance formula is derived for the symmetric scheme; for the
    asymmetric one the mean must still be 1."""
    N, L, M = 64, 4, 20000
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1 / np.sqrt(N), size=(M, L, N))
    b = rng.normal(0, 1 / np.sqrt(N), size=(M, L, N))
    h = HRR(asymmetric=asym)
    S = h.bind(a, b).sum(axis=1)
    ov = np.einsum("mn,mn->m", h.unbind(S, b[:, 0]), a[:, 0])
    se = ov.std() / np.sqrt(M)
    assert abs(ov.mean() - 1.0) < 4 * se
    if not asym:
        var = theory.hrr_overlap_variance(N, L, "correct")
        # sample variance has relative s.e. ~ sqrt(2/M) for near-Gaussian data
        assert abs(ov.var() / var - 1.0) < 4 * np.sqrt(2 / M) * 1.5


def test_binarized_snr_leading_order_SI_eq_34():
    """SNR from sgn(S) ~ (2/pi) N/L to leading order in L/N (SI Eq. 34)."""
    N, L, M = 2048, 8, 3000
    rng = np.random.default_rng(3)
    h = HRR(asymmetric=False)
    a = rng.normal(0, 1 / np.sqrt(N), size=(M, L, N))
    b = rng.normal(0, 1 / np.sqrt(N), size=(L, N))
    sigma = binarize(h.bind(a, b[None]).sum(axis=1))
    a_hat = h.unbind(sigma, b[0][None])
    sig = np.einsum("mn,mn->m", a_hat, a[:, 0])
    other = np.einsum("mn,mn->m", a_hat, a[:, 1])        # an independent-enough item
    snr = sig.mean() ** 2 / np.mean(other ** 2)
    # leading-order formula; corrections are O(1/L) and O(L/N), so allow 15%
    assert abs(snr / theory.snr0(N, L, binarized=True) - 1) < 0.15


def test_sequence_query_returns_successor_not_predecessor():
    N = 4096
    bank = ItemBank(N, 5, np.random.SeedSequence(0))
    seqs = Sequences(bank, np.arange(5)[None, :], HRR(asymmetric=True))
    S = seqs.encode(4)[0]
    items = bank.get(np.arange(5))
    ov = items @ HRR(asymmetric=True).unbind(S, items[2])
    assert np.argmax(ov) == 3
    assert abs(ov[1]) < 0.2 and ov[3] > 0.7


def test_item_bank_is_deterministic_and_N01_over_N():
    bank = ItemBank(50, 1000, np.random.SeedSequence(7), block=64)
    ids = np.array([3, 999, 64, 3])
    x = bank.get(ids)
    np.testing.assert_array_equal(x, bank.get(ids))
    np.testing.assert_array_equal(x[0], x[3])
    allv = np.concatenate([blk for _, blk in bank.iter_blocks(1000)])
    np.testing.assert_array_equal(allv[ids], x)
    assert abs(allv.var() * 50 - 1) < 0.02 and abs(allv.mean()) < 0.01


def test_unused_items_match_explicit_items_in_distribution():
    """Clean-up with 990 unused items sampled analytically must give the same
    error rate and noise as a Dictionary with all 1000 items explicit."""
    N, M, D = 32, 4000, 1000
    rng = np.random.default_rng(5)
    bank = ItemBank(N, D, np.random.SeedSequence(11))
    true = rng.integers(0, 10, size=M)
    a_hat = 0.55 * bank.get(true) + rng.normal(0, 1 / np.sqrt(N), size=(M, N))
    full = Dictionary(bank, D, D).readout(a_hat, true, rng)
    part = Dictionary(bank, 10, D).readout(a_hat, true, rng)
    p1, p2 = full.error.mean(), part.error.mean()
    se = np.sqrt(p1 * (1 - p1) / M + p2 * (1 - p2) / M)
    assert 0.1 < p1 < 0.9 and abs(p1 - p2) < 4 * se
    np.testing.assert_array_equal(full.signal, part.signal)
    expected = np.sum(a_hat ** 2, axis=1) / N            # E[(d . a_hat)^2 | a_hat]
    assert abs(part.noise_sq.mean() / expected.mean() - 1) < 0.02


def test_unused_max_law():
    """max of n iid N(0, s^2): P(max <= x) = Phi(x/s)^n, KS test."""
    N, n = 16, 5000
    bank = ItemBank(N, 1, np.random.SeedSequence(0))
    dic = Dictionary(bank, 1, n + 1)
    a_hat = np.ones((20000, N)) * 0.5
    mx, _ = dic.unused_overlaps(a_hat, np.random.default_rng(6))
    s = np.linalg.norm(a_hat[0]) / np.sqrt(N)
    p = stats.kstest(mx / s, lambda x: stats.norm.cdf(x) ** n).pvalue
    assert p > 1e-3


def test_structure_encode_matches_loop_and_prefix():
    N, P, L = 31, 3, 5
    bank = ItemBank(N, P * L, np.random.SeedSequence(1))
    attrs = np.random.default_rng(2).normal(size=(L, N))
    st = Structures(bank, np.arange(P * L).reshape(P, L), attrs, HRR(False), chunk=2)
    enc = st.encode([L, 2])
    for mu in range(P):
        objs = bank.get(np.arange(mu * L, mu * L + L))
        full = sum(_loop_conv(objs[l], attrs[l]) for l in range(L))
        cue = sum(_loop_conv(objs[l], attrs[l]) for l in range(2))
        np.testing.assert_allclose(enc[L][mu], full, atol=1e-10)
        np.testing.assert_allclose(enc[2][mu], cue, atol=1e-10)


def test_cue_overlap_matches_eq_30():
    """m0 = (2/pi) arctan sqrt(x/(1-x)), x = L0/L: the arcsine law for sgn of
    two Gaussians with correlation sqrt(L0/L)."""
    N, P, L = 4096, 40, 12
    bank = ItemBank(N, P * L, np.random.SeedSequence(3))
    attrs = np.random.default_rng(4).normal(0, 1 / np.sqrt(N), size=(L, N))
    st = Structures(bank, np.arange(P * L).reshape(P, L), attrs, HRR(False))
    for L0 in (1, 3, 6):
        enc = st.encode([L, L0])
        m = np.mean(binarize(enc[L]) * binarize(enc[L0]))
        assert abs(m - theory.m0_of_cue(L0, L)) < 0.01


def test_shared_run_construction():
    from experiments.sequences import item_ids
    ids = item_ids(3, 6, 2, 4, np.random.default_rng(0))
    np.testing.assert_array_equal(ids, [[0, 1, 2, 3, 4, 5],
                                        [6, 7, 8, 3, 4, 11],
                                        [12, 13, 14, 3, 4, 17]])
    rnd = item_ids(50, 10, 3, None, np.random.default_rng(1))
    for mu in range(1, 50):
        same = np.flatnonzero(rnd[mu] == rnd[mu - 1])
        assert len(same) == 3 and np.all(np.diff(same) == 1) and same[0] >= 2


def test_cleanup_unused_winner_matches_explicit_dictionary():
    """P(an item outside the first 10 wins) must agree between a fully explicit
    Dictionary and one whose other items are sampled analytically."""
    N, M, D = 32, 4000, 1000
    rng = np.random.default_rng(8)
    bank = ItemBank(N, D, np.random.SeedSequence(12))
    true = rng.integers(0, 10, size=M)
    a_hat = 0.55 * bank.get(true) + rng.normal(0, 1 / np.sqrt(N), size=(M, N))
    ids_full, _ = Dictionary(bank, D, D).cleanup(a_hat, rng)
    ids_part, vec = Dictionary(bank, 10, D).cleanup(a_hat, rng)
    p1, p2 = np.mean(ids_full >= 10), np.mean(ids_part == -1)
    se = np.sqrt(p1 * (1 - p1) / M + p2 * (1 - p2) / M)
    assert 0.1 < p1 < 0.9 and abs(p1 - p2) < 4 * se
    known = ids_part >= 0
    np.testing.assert_array_equal(vec[known], bank.get(ids_part[known]))
    # a sampled winner beats every explicit item and has norm^2 = x^2/|a|^2 + (N-1)/N on average
    ov = np.einsum("bn,bn->b", vec, a_hat)
    expl = a_hat @ bank.get(np.arange(10)).T
    assert np.all(ov[~known] > expl[~known].max(axis=1))
    n2 = a_hat[~known] ** 2
    resid = np.sum(vec[~known] ** 2, axis=1) - ov[~known] ** 2 / n2.sum(axis=1)
    assert abs(resid.mean() - (N - 1) / N) < 4 * np.sqrt(2 * (N - 1)) / N / np.sqrt((~known).sum())
