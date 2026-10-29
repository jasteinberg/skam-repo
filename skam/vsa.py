"""Vector-symbolic layer: binding, the Dictionary, structures and sequences.

Everything that turns items into distributed vectors and back lives here.

Binding (Eqs. 2-4). ``Binder`` is the interface: g(a, b) = bind(a, b) and
a_hat = unbind(S_hat, b). Structures and sequences use only this interface, so
any quadratic binding g_k(a, b) = a^T G^k b with its unbinding (Eqs. 2, 4) can
be supplied by subclassing ``Binder``. ``HRR`` is the scheme used in the paper
(Eqs. 15-18); it binds by circular convolution and unbinds by circular
correlation,

    g_k(a, b) = sum_j a_j b_{k-j}                          (Eq. 15)
    a_hat_k   = sum_j b_j S_{j+k}                          (Eq. 16, see note)

Note on Eq. 16. As printed, a_hat_k = sum_j S_j b_{j+k}, the shift sits on b.
Substituting Eq. 15 gives a_hat_k = a_{-k} sum_j b_j^2 + noise, i.e. the
involution of a, which has zero overlap with a. Plate's correlation puts the
shift on S, a_hat_k = sum_j b_j S_{j+k}, and returns a_k sum_j b_j^2 + noise as
SI Eq. 8 states. We implement Plate's form. In Fourier space

    bind:   G = A B          unbind:  A_hat = conj(B) S

Asymmetric binding for sequences (Eqs. 10-11) exchanges the two operations,

    bind:   G = conj(B) A    unbind:  A_hat = B S

so g(a, b) != g(b, a) and unbinding with b still returns a.

Items (Methods, Simulations). Objects and attributes are i.i.d. N(0, 1/N)
vectors. A Dictionary of size D stores all familiar objects; clean-up is the ML
read-out a_tilde = argmax_d d . a_hat (Eq. 5).

Exact treatment of unused Dictionary items. Items that appear in no stored
structure are independent of every network state and every estimator. For such
an item d and any estimator a_hat, d . a_hat | a_hat ~ N(0, |a_hat|^2 / N)
exactly, and the overlaps of different unused items are independent given
a_hat. We therefore never materialize them: the maximum of n such overlaps is
sampled from its exact law, P(max <= x) = Phi(x / s)^n, and their sum of
squares from s^2 chi^2_n. Items that are used in some structure are generated
block-wise from a seed so that a D x N matrix is never held in memory.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri

from . import backend

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------
class Binder(ABC):
    """A binding scheme (Eqs. 2, 4). Methods broadcast over leading axes; the
    last axis has length N. ``asymmetric`` states whether g(a, b) != g(b, a),
    which sequences require (Eq. 11)."""
    asymmetric: bool

    @abstractmethod
    def bind(self, a: FloatArray, b: FloatArray) -> FloatArray:
        """g(a, b)."""

    @abstractmethod
    def unbind(self, s: FloatArray, b: FloatArray) -> FloatArray:
        """Estimate of the partner of ``b`` in the superposition ``s``."""

    def prefix_sums(self, A: FloatArray, B: FloatArray, prefixes: list[int]
                    ) -> dict[int, FloatArray]:
        """{n: sum_{l < n} g(A[..., l, :], B[..., l, :])} for each n in ``prefixes``.

        This is the encoding of Eqs. 3 and 6. The default binds each pair and
        accumulates; a subclass may override it with a faster equivalent.
        """
        shape = np.broadcast_shapes(np.shape(A), np.shape(B))
        acc = np.zeros(shape[:-2] + shape[-1:])
        out = {}
        if 0 in prefixes:
            out[0] = acc.copy()
        for l in range(shape[-2]):
            acc = acc + self.bind(A[..., l, :], B[..., l, :])
            if l + 1 in prefixes:
                out[l + 1] = acc.copy()
        return out


class HRR(Binder):
    """Holographic reduced representation binding (Eqs. 15-18).

    ``asymmetric=False``: bind = circular convolution, unbind = correlation.
    ``asymmetric=True``: the two are exchanged (sequences, Eqs. 10-11).
    """

    def __init__(self, asymmetric: bool):
        self.asymmetric = asymmetric

    @staticmethod
    def spectrum(x: FloatArray) -> NDArray[np.complex128]:
        return np.fft.rfft(np.asarray(x, dtype=np.float64), axis=-1)

    @staticmethod
    def inverse(X: NDArray[np.complex128], N: int) -> FloatArray:
        return np.fft.irfft(X, n=N, axis=-1)

    def bind_spectrum(self, A: NDArray[np.complex128],
                      B: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Spectrum of g(a, b) from the spectra of a and b."""
        return np.conj(B) * A if self.asymmetric else A * B

    def bind(self, a: FloatArray, b: FloatArray) -> FloatArray:
        """g(a, b), Eq. 15 (or its asymmetric counterpart)."""
        N = np.shape(a)[-1]
        return self.inverse(self.bind_spectrum(self.spectrum(a), self.spectrum(b)), N)

    def unbind(self, s: FloatArray, b: FloatArray) -> FloatArray:
        """a_hat from a (super)position ``s`` queried with ``b``, Eqs. 4 and 16."""
        N = np.shape(s)[-1]
        S, B = self.spectrum(s), self.spectrum(b)
        return self.inverse(B * S if self.asymmetric else np.conj(B) * S, N)

    def prefix_sums(self, A: FloatArray, B: FloatArray, prefixes: list[int]
                    ) -> dict[int, FloatArray]:
        """Same as Binder.prefix_sums: a sum of bindings is the inverse
        transform of the summed spectral products, one FFT per prefix."""
        N = np.shape(A)[-1]
        cums = np.cumsum(self.bind_spectrum(self.spectrum(A), self.spectrum(B)), axis=-2)
        return {n: (self.inverse(cums[..., n - 1, :], N) if n > 0
                    else np.zeros(cums.shape[:-2] + (N,)))
                for n in prefixes}


def binarize(s: FloatArray) -> FloatArray:
    """sigma = sgn(S_hat). S_hat_i = 0 has probability zero; it is mapped to +1."""
    sigma = np.sign(s)
    sigma[sigma == 0] = 1.0
    return sigma


# ---------------------------------------------------------------------------
# Item bank and Dictionary
# ---------------------------------------------------------------------------
class ItemBank:
    """``size`` i.i.d. N(0, 1/N) vectors, generated block-wise from a seed.

    Block b is always produced by the same generator, so ``get`` returns the
    same vector for the same id on every call without storing the bank.
    """

    def __init__(self, N: int, size: int, seed: np.random.SeedSequence, block: int = 2048):
        if size < 0 or N <= 0 or block <= 0:
            raise ValueError(f"bad ItemBank sizes N={N} size={size} block={block}")
        self.N = N
        self.size = size
        self.block_size = block
        self._entropy = seed.entropy
        self._key = tuple(seed.spawn_key)

    def block(self, b: int) -> FloatArray:
        start = b * self.block_size
        if not 0 <= start < self.size:
            raise IndexError(f"block {b} outside bank of size {self.size}")
        n = min(self.block_size, self.size - start)
        ss = np.random.SeedSequence(self._entropy, spawn_key=self._key + (b,))
        return np.random.default_rng(ss).normal(0.0, 1.0 / np.sqrt(self.N), size=(n, self.N))

    def iter_blocks(self, stop: int) -> Iterator[tuple[int, FloatArray]]:
        """Yield (start, items[start:start+block]) covering ids [0, stop)."""
        if stop > self.size:
            raise IndexError(f"stop {stop} > bank size {self.size}")
        for b in range((stop + self.block_size - 1) // self.block_size):
            start = b * self.block_size
            items = self.block(b)
            yield start, items[: stop - start]

    def get(self, ids: IntArray) -> FloatArray:
        """Items for an integer array of ids, any shape; returns shape ids.shape + (N,)."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.size):
            raise IndexError("item id outside bank")
        flat = ids.ravel()
        out = np.empty((flat.size, self.N))
        blocks = flat // self.block_size
        for b in np.unique(blocks):
            sel = blocks == b
            out[sel] = self.block(int(b))[flat[sel] - b * self.block_size]
        return out.reshape(ids.shape + (self.N,))


@dataclass(frozen=True)
class Readout:
    """Clean-up of a batch of estimators against a Dictionary.

    error:     argmax_d d . a_hat != true item                       (Eq. 5)
    signal:    a_hat . a_true
    noise_sq:  mean over the D-1 incorrect items of (a_hat . a_d')^2
    The SNR of Eq. 20 is mean(signal)^2 / mean(noise_sq) over a population.
    """
    error: BoolArray
    signal: FloatArray
    noise_sq: FloatArray


class Dictionary:
    """Clean-up memory of D familiar objects (Eq. 5).

    Item ids [0, n_explicit) are taken from ``bank`` and may appear in stored
    structures; the remaining D - n_explicit items appear in none and are
    treated exactly as described in the module docstring.
    """

    def __init__(self, bank: ItemBank, n_explicit: int, D: int):
        if not 0 < n_explicit <= bank.size:
            raise ValueError(f"n_explicit={n_explicit} must be in (0, {bank.size}]")
        if D < n_explicit:
            raise ValueError(f"D={D} smaller than the {n_explicit} items in use")
        self.bank = bank
        self.n_explicit = n_explicit
        self.D = D

    @property
    def N(self) -> int:
        return self.bank.N

    @property
    def n_unused(self) -> int:
        return self.D - self.n_explicit

    def explicit_overlaps(self, a_hat: FloatArray, true_ids: IntArray | None
                          ) -> tuple[FloatArray, IntArray, FloatArray, FloatArray]:
        """One pass over the explicit items, on the active backend.

        Returns (best overlap, its id, overlap with the true item, sum of
        squared overlaps). With ``true_ids`` the true item is excluded from the
        best and from the sum; without, every explicit item counts and the
        signal is NaN.
        """
        B = a_hat.shape[0]
        best = np.full(B, -np.inf)
        best_id = np.full(B, -1, dtype=np.int64)
        signal = np.full(B, np.nan)
        sumsq = np.zeros(B)
        kernel = _block_scan_torch(a_hat) if backend.name() == "torch" else _block_scan_numpy(a_hat)
        for start, items in self.bank.iter_blocks(self.n_explicit):
            local = None if true_ids is None else true_ids - start
            cand, j, sig, hit, ssq = kernel(items, local)
            better = cand > best
            best[better] = cand[better]
            best_id[better] = start + j[better]
            signal[hit] = sig
            sumsq += ssq
        if true_ids is not None and np.isnan(signal).any():
            raise ValueError("a true id lies outside the explicit items")
        return best, best_id, signal, sumsq

    def unused_overlaps(self, a_hat: FloatArray, rng: np.random.Generator
                        ) -> tuple[FloatArray, FloatArray]:
        """Exact samples of (max, sum of squares) of the overlaps of a_hat
        with the n_unused items that appear in no structure."""
        n = self.n_unused
        B = a_hat.shape[0]
        if n == 0:
            return np.full(B, -np.inf), np.zeros(B)
        scale = np.linalg.norm(a_hat, axis=1) / np.sqrt(self.N)
        # max of n iid N(0,1): Phi(x)^n = U  ->  upper tail q = 1 - U^(1/n)
        q = -np.expm1(np.log(rng.random(B)) / n)
        mx = -ndtri(q) * scale
        sumsq = rng.chisquare(n, size=B) * scale ** 2
        return mx, sumsq

    def readout(self, a_hat: FloatArray, true_ids: IntArray, rng: np.random.Generator) -> Readout:
        """ML clean-up of each row of ``a_hat`` (B, N) against all D items."""
        a_hat = np.atleast_2d(np.asarray(a_hat, dtype=np.float64))
        true_ids = np.asarray(true_ids, dtype=np.int64)
        if true_ids.shape != (a_hat.shape[0],):
            raise ValueError(f"true_ids shape {true_ids.shape} != ({a_hat.shape[0]},)")
        best, _, signal, sumsq = self.explicit_overlaps(a_hat, true_ids)
        umax, usumsq = self.unused_overlaps(a_hat, rng)
        error = np.maximum(best, umax) >= signal
        return Readout(error=error, signal=signal, noise_sq=(sumsq + usumsq) / (self.D - 1))

    def cleanup(self, a_hat: FloatArray, rng: np.random.Generator) -> tuple[IntArray, FloatArray]:
        """ML clean-up a_tilde = argmax_d d . a_hat over all D items (Eq. 5).

        Returns the winner's id and its vector, for feeding the decoded item
        forward. If one of the unused items wins, its id is -1 and its vector
        is sampled from its exact conditional law: an isotropic N(0, I/N) item
        whose overlap with a_hat is the winning value x is
            d = (x / |a_hat|^2) a_hat + z_perp,
        with z_perp ~ N(0, I/N) projected orthogonally to a_hat.
        """
        a_hat = np.atleast_2d(np.asarray(a_hat, dtype=np.float64))
        best, arg, _, _ = self.explicit_overlaps(a_hat, None)
        umax, _ = self.unused_overlaps(a_hat, rng)
        unused = umax > best
        arg[unused] = -1
        vectors = np.empty_like(a_hat)
        known = ~unused
        vectors[known] = self.bank.get(arg[known])
        if unused.any():
            u = a_hat[unused]
            norm2 = np.sum(u * u, axis=1, keepdims=True)
            z = rng.normal(0.0, 1.0 / np.sqrt(self.N), size=u.shape)
            z -= np.sum(z * u, axis=1, keepdims=True) / norm2 * u
            vectors[unused] = umax[unused][:, None] / norm2 * u + z
        return arg, vectors


BlockScan = Callable[[FloatArray, "IntArray | None"],
                     tuple[FloatArray, IntArray, FloatArray, BoolArray, FloatArray]]


def _block_scan_numpy(a_hat: FloatArray) -> BlockScan:
    """Kernel for one block of items: (max overlap, argmax, overlaps with the
    true items that fall in the block, mask of those rows, sum of squares),
    excluding the true item when ``local`` (true id - block start) is given."""
    rows = np.arange(a_hat.shape[0])

    def scan(items: FloatArray, local: IntArray | None
             ) -> tuple[FloatArray, IntArray, FloatArray, BoolArray, FloatArray]:
        ov = a_hat @ items.T
        if local is None:
            hit = np.zeros(len(rows), dtype=bool)
        else:
            hit = (local >= 0) & (local < items.shape[0])
        sig = ov[rows[hit], local[hit]] if local is not None else np.empty(0)
        if local is not None:
            ov[rows[hit], local[hit]] = -np.inf
        j = np.argmax(ov, axis=1)
        cand = ov[rows, j]
        if local is not None:
            ov[rows[hit], local[hit]] = 0.0
        return cand, j, sig, hit, np.sum(ov * ov, axis=1)
    return scan


def _block_scan_torch(a_hat: FloatArray) -> BlockScan:
    """The same kernel in float32 on the torch device."""
    torch = backend.torch_module()
    dev = backend.device()
    a = torch.as_tensor(a_hat.astype(np.float32), device=dev)
    rows = torch.arange(a.shape[0], device=dev)

    def scan(items: FloatArray, local: IntArray | None
             ) -> tuple[FloatArray, IntArray, FloatArray, BoolArray, FloatArray]:
        ov = a @ torch.as_tensor(items.astype(np.float32), device=dev).T
        if local is None:
            hit = torch.zeros(a.shape[0], dtype=torch.bool, device=dev)
            sig = torch.empty(0, device=dev)
        else:
            loc = torch.as_tensor(local, device=dev)
            hit = (loc >= 0) & (loc < items.shape[0])
            r, c = rows[hit], loc[hit]
            sig = ov[r, c].clone()
            ov[r, c] = -float("inf")
        cand, j = ov.max(dim=1)
        if local is not None:
            ov[r, c] = 0.0
        ssq = (ov * ov).sum(dim=1)
        return (cand.cpu().numpy().astype(np.float64), j.cpu().numpy().astype(np.int64),
                sig.cpu().numpy().astype(np.float64), hit.cpu().numpy(), ssq.cpu().numpy().astype(np.float64))
    return scan


# ---------------------------------------------------------------------------
# Structures (Eqs. 1, 3, 6) and sequences (Eqs. 10, 11)
# ---------------------------------------------------------------------------
class Structures:
    """P structures of L object/attribute relations (Eq. 1).

    ``object_ids[mu, l]`` indexes ``bank``; ``attributes`` is (L, N), shared by
    all structures, b_l^mu = b_l (Methods, Simulations).
    """

    def __init__(self, bank: ItemBank, object_ids: IntArray, attributes: FloatArray,
                 binder: Binder, chunk: int = 256):
        object_ids = np.asarray(object_ids, dtype=np.int64)
        if object_ids.ndim != 2 or attributes.shape != (object_ids.shape[1], bank.N):
            raise ValueError(f"object_ids {object_ids.shape} vs attributes {attributes.shape}")
        self.bank = bank
        self.object_ids = object_ids
        self.attributes = attributes
        self.binder = binder
        self.chunk = chunk

    @property
    def P(self) -> int:
        return self.object_ids.shape[0]

    @property
    def L(self) -> int:
        return self.object_ids.shape[1]

    @property
    def N(self) -> int:
        return self.bank.N

    def encode(self, prefixes: list[int]) -> dict[int, FloatArray]:
        """S_hat over the first n relations for each n in ``prefixes`` (Eqs. 3, 6).

        Returns {n: (P, N)}. n = L is the full structure, n = L0 the cue.
        """
        if any(not 0 <= n <= self.L for n in prefixes):
            raise ValueError(f"prefixes {prefixes} outside [0, {self.L}]")
        out = {n: np.empty((self.P, self.N)) for n in prefixes}
        for s in range(0, self.P, self.chunk):
            objects = self.bank.get(self.object_ids[s:s + self.chunk])   # (c, L, N)
            sums = self.binder.prefix_sums(objects, self.attributes[None], prefixes)
            for n in prefixes:
                out[n][s:s + self.chunk] = sums[n]
        return out

    def unbind(self, states: FloatArray, l: int) -> FloatArray:
        """a_hat for relation ``l`` of every structure: query with b_l (Eq. 4)."""
        return self.binder.unbind(states, self.attributes[l][None, :])


class Sequences:
    """P temporal sequences of L events (Eq. 10), ``item_ids`` (P, L) into ``bank``.

    Encoded with asymmetric binding as S_hat = sum_l g(a_{l+1}, a_l) (Eq. 11):
    each relation is (object = successor, attribute = predecessor), so querying
    with a_l returns a_{l+1} and the sequence unfolds forward from a_1.
    Eq. 11 lists the pair as g(a_l, a_{l+1}); the order of the two arguments is
    a naming convention once binding is asymmetric.
    """

    def __init__(self, bank: ItemBank, item_ids: IntArray, binder: Binder):
        if not binder.asymmetric:
            raise ValueError("sequences require asymmetric binding (Eq. 11)")
        self.bank = bank
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        self.binder = binder

    @property
    def P(self) -> int:
        return self.item_ids.shape[0]

    @property
    def L(self) -> int:
        return self.item_ids.shape[1]

    def encode(self, n_relations: int) -> FloatArray:
        """Sum of the first ``n_relations`` successor bindings, (P, N)."""
        if not 1 <= n_relations <= self.L - 1:
            raise ValueError(f"n_relations={n_relations} outside [1, {self.L - 1}]")
        items = self.bank.get(self.item_ids[:, : n_relations + 1])   # (P, n+1, N)
        return self.binder.bind(items[:, 1:], items[:, :-1]).sum(axis=1)

    def unfold(self, states: FloatArray, dictionary: Dictionary,
               rng: np.random.Generator) -> tuple[IntArray, FloatArray]:
        """Decode every sequence from its retrieved state (Fig. 6a).

        Start from the known first event a_1, unbind to get a_hat_2, clean up,
        and query with the cleaned-up item for the next position, so an error
        propagates down the chain. Returns decoded ids (P, L) (column 0 is a_1;
        -1 marks an item that is in no stored sequence) and the estimators
        (P, L, N) (column 0 unused, zeros).
        """
        P, L = self.P, self.L
        decoded = np.empty((P, L), dtype=np.int64)
        a_hats = np.zeros((P, L, self.bank.N))
        decoded[:, 0] = self.item_ids[:, 0]
        query = self.bank.get(decoded[:, 0])
        for l in range(1, L):
            a_hats[:, l] = self.binder.unbind(states, query)
            decoded[:, l], query = dictionary.cleanup(a_hats[:, l], rng)
        return decoded, a_hats
