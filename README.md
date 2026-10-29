# skam: associative memory of structured knowledge

Code that reproduces every data figure of

> J. Steinberg and H. Sompolinsky, *Associative memory of structured knowledge*,
> Scientific Reports **12**, 21808 (2022). Open access:
> https://doi.org/10.1038/s41598-022-25708-y

including the Supplementary Material. The implementation is written from the
equations and Methods of the paper.

<small>*Drafted with the assistance of Claude (Anthropic).*</small>

## The model in brief

A structure is a set of $L$ object/attribute pairs, $S = \{(a_1,b_1),\dots,(a_L,b_L)\}$,
with $a, b \in \mathbb{R}^N$ drawn i.i.d. from $\mathcal N(0, 1/N)$. Each pair is bound by
circular convolution and the bound pairs are summed,

$$\hat S = \sum_{\ell=1}^{L} a_\ell \circledast b_\ell .$$

The binarized vectors $\sigma^\mu = \mathrm{sgn}(\hat S^\mu)$ of $P = \alpha N$ structures
are stored as fixed points of a Hopfield network $\sigma_i(t) = \mathrm{sgn}\big(\sum_j J_{ij}\sigma_j(t-1)\big)$,
with $J$ given by the pseudo-inverse, Hebb or Storkey rule. A structure is
retrieved from the network by creating a retrieval cue binarized sum of $L_0 < L$ of its pairs. A relation that was
not in the cue is then read out by unbinding with its attribute, $\hat a_\ell = \sigma \star b_\ell$
(circular correlation), and cleaning up against a Dictionary of $D$ items, $\tilde a = \arg\max_{d} \, d\cdot \hat a$.

The decoding error is set by the signal-to-noise ratio
$\mathrm{SNR} = \langle \hat a \cdot a_d\rangle^2 / \langle (\hat a \cdot a_{d'})^2\rangle$ through Eq. 19 of the paper. The central result is that this
read-out remains accurate when the cue lies outside the basin of attraction of
the stored structure, because the network still relaxes to a state with large
overlap with it.

## Installation

The base install needs only numpy, scipy, matplotlib and threadpoolctl.

```bash
pip install -r requirements.txt
python -m pytest
```

PyTorch is optional and is used only for GPU runs (see below): `pip install torch`, or `pip install ".[gpu]"` to install the `skam` package with it. The `experiments` and `figures` modules are run from the repository root.

## Reproducing the figures

Each figure is produced in two steps. The first runs the simulations and the
second draws the figure from the saved trials.

```bash
python -m figures run  fig3 --scale quick --seed 0 --workers 8
python -m figures plot fig3 --scale quick --seed 0
```

The available figures are `fig1`, `fig3`, `fig4`, `fig5`, `fig6` and `si` (all
eight Supplementary figures), or `all`. Two scales are defined for every
figure. `--scale paper` uses the parameters stated in the paper (network sizes
up to $N = 16000$ and the published numbers of trials). `--scale quick` draws
the same panels at smaller $N$ with about ten trials per point, which is enough
to check that everything works. `python -m figures list all --scale paper`
prints the number of trials a run needs. Figures are written to
`figures/out/<scale>/`.

To draw the figures one at a time instead, open `notebooks/figures.ipynb`. Each
cell draws one figure inline from the saved trials, selected by
`Ctx("quick", seed=0)`. In Python, `figures.fig3.plot(ctx)` returns the figures
as a dictionary of matplotlib `Figure` objects.

| Paper | Module | Simulation |
|---|---|---|
| Fig. 1b, c | `figures/fig1.py` | `experiments/unbinding.py` |
| Fig. 3a to c | `figures/fig3.py` | `experiments/structures.py` |
| Fig. 4b | `figures/fig4.py` | `experiments/random_patterns.py` |
| Fig. 5 | `figures/fig5.py` | `experiments/structures.py`, with $f(\alpha)$ from Fig. 4b |
| Fig. 6b, e, f | `figures/fig6.py` | `experiments/sequences.py` |
| Fig. 6d | `figures/fig6.py` | `experiments/ks_sequences.py` |
| Supplementary Figs. 1 to 8 | `figures/si.py` | `structures`, `random_patterns` |

The schematics (Figs. 1a, 2, 4a, 6a and 6c) are not reproduced.

## Layout

```
skam/
  vsa.py        HRR binding, the Dictionary and clean-up, structures, sequences   Eqs. 1-6, 10-11, 15-18
  memory.py     pseudo-inverse, Hebb and Storkey rules; parallel, serial and
                finite-temperature dynamics                                       Eqs. 23-28, SI Eq. 66
  measures.py   overlaps m, r, m^0; SNR; shape of p(m); basin size m_min          Eqs. 20, 29, 32; SI Eqs. 60-62
  theory.py     analytic predictions                                              Eqs. 9, 19, 22, 30, 31; SI Eqs. 20, 34, 89
  ks.py         sequences of attractors                                           SI Eqs. 85-89
  backend.py    numpy (default) or torch for the dense hot loops
  trials.py     parallel, resumable execution of independent trials
experiments/    one trial function per kind of simulation
figures/        one module per figure: parameter grids and plotting
notebooks/      figures.ipynb draws each figure inline
tests/          tests against closed forms, explicit loops and exact invariances
```

## Trials, seeds and parallelism

A trial is a pure function of a parameter point and a seed, and each trial is
saved to its own file under `results/`. The seed of trial $k$ is derived from
the base seed, a hash of the parameter point and $k$, so results do not depend
on the number of workers or on the order in which trials run. The file location
includes a hash of the source code, so trials computed by different versions of
the code are never combined. Rerunning a command only computes the trials that
are missing. Each simulation declares its parameters, and a parameter that is
missing or unknown raises an error rather than being ignored. Each trial file
also records the Python, numpy and scipy versions that produced it, and
`plot` refuses to combine trials from different versions.

**Memory.** Each worker holds one network. At the largest paper-scale points
(dense Storkey couplings at $N = 8000$, random patterns at $N = 16000$) this is
2 to 3 GB, so choose `--workers` such that the workers fit in RAM.

**Sharding.** To split a run across machines, run the same command at the same
commit on machine $i = 0,\dots,n-1$ with `--shard i/n`, then copy the `results/`
directories into one before plotting. The union of the shards is identical to
an unsharded run.

**GPU.** With PyTorch installed, `--backend torch` runs the parallel dynamics
and the Dictionary overlaps on a GPU in float32 (`--device cuda` or `mps`,
chosen automatically if omitted). Use one or two workers, since the workers
share the GPU. Item generation, encoding, the pseudo-inverse solve (in float64)
and serial dynamics stay in numpy. At $N = 8000$, $P = 1600$, $L = 30$ one
retrieval of all structures took 67 s on one CPU core and 3.7 s on Apple MPS,
with identical decoding error. In float32 a local field within about $10^{-7}$ of
zero can change sign, so the two backends agree statistically rather than bit
for bit. Results are stored separately per backend, and `plot` takes the same
`--backend` flag.

## Implementation notes

**Binding schemes.** Structures and sequences use binding only through the
`Binder` interface in `skam/vsa.py` (`bind`, `unbind`). HRR is the scheme of the
paper. Another quadratic binding $g_k(a,b) = a^\top G^k b$ (Eq. 2) is added by
subclassing `Binder`, without changing the rest of the code.

The following are exact reformulations, not approximations.

- **Pseudo-inverse couplings.** $J = \frac1N \sigma^\top C^{-1}\sigma$ with $C = \frac1N \sigma\sigma^\top$ is
  kept in factored form. $C^{-1}\sigma$ is obtained from a Cholesky factorization of the
  $P\times P$ matrix $C$, and the field of a batch of $B$ states costs $O(BPN)$. No
  $N\times N$ matrix is formed. Learning costs $O(P^2N + P^3)$, roughly a hundred times less
  than retrieval of all $P$ structures.
- **Dictionary items that appear in no stored structure.** Such an item is
  independent of every network state, so its overlap with an estimator is
  exactly $\mathcal N(0, |\hat a|^2/N)$. The maximum of $n$ such overlaps is sampled from its
  law $\Phi(x)^n$ and their sum of squares from $\chi^2_n$. When one of these items wins the
  clean-up of a sequence, its vector is drawn from its conditional law,
  $d = (x/|\hat a|^2)\,\hat a + z_\perp$. A Dictionary of $D = 30N$ items therefore costs no more
  than the items in use.
- **Items in use** are regenerated block by block from a seed, so a $D \times N$ array is
  never held in memory.

## Additional details

- **Eq. 16.** As printed, $\hat a_k = \sum_j \hat S_j b_{j+k}$ returns the involution $a_{-k}$, which has no overlap with $a$. The code uses Plate's correlation $\hat a_k = \sum_j b_j \hat S_{j+k}$, which
  inverts Eq. 15 and gives the moments of SI Eqs. 19 and 20.
- **Fig. 1** is computed from binarized structures. The published
  $\mathrm{SNR}^{-1} = 0.095$ at $N = 500$, $L = 30$ equals $(\pi/2)L/N$ (SI Eq. 34).
- **Storkey rule.** Eq. 27 is implemented as printed in Methods, including the
  factor $(N+1)/(N-1)$.
- **$m_{\min}(\alpha)$** is the smallest $m_0$ at which $\langle m\rangle \ge 0.99$, measured against the fixed
  point that the stored pattern itself reaches. This reads "the minimal overlap
  required for convergence to the correct fixed point on average" literally and
  reproduces Supplementary Fig. 1a. $l_c$ uses the same criterion in $L_0/L$ and is
  compared with Eq. 31. The pseudo-inverse rule is used only for $\alpha < 1$; at $\alpha = 1$ the
  couplings vanish.
- **$f(\alpha)$** (Eq. 7) is the slope through the origin of $m^*$ against $m_0$, fitted on
  $m_0 \in \{0.01,\dots,0.05\}$ where no cue reaches the memory.
- **$\sigma_m^2$** is the variance of the $m<1$ mode, as defined in Methods.
- **Cues for random patterns** flip exactly $\mathrm{round}(N(1-m_0)/2)$ spins.
- **Serial dynamics.** One time step is a sweep over all sites in a fresh
  random order. A zero local field leaves the spin unchanged.
- **Sequences.** Relations are bound asymmetrically by exchanging convolution and
  correlation. The cue is the first relation, and each cleaned-up item queries
  the next. The Dictionary has $D = 30N$ items as in Methods. Neighbouring
  sequences share a contiguous run of $L_s$ events at the same positions, which
  gives $L_s - 1$ shared relations.
- **SI Eq. 88.** The kernel $w(t) = \tau^{-1}\Theta(t-\tau)$ has a divergent integral. The code
  uses the window $w(t) = \tau^{-1}\Theta(t)\,\Theta(\tau - t)$, which gives the period $t_0 = \frac{\tau}{2}(1+\lambda^{-1})$ of
  SI Eq. 89.
- **Not stated in the paper** and chosen here: for Fig. 6d, sequences of length
  15 sharing their seventh pattern with a pseudo-inverse $J^{(1)}$. For Fig. 6e, $L = 14$
  with the shared event at position 7.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
