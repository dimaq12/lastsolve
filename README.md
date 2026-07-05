# lastsolve

![lastsolve — the last few solves you'll ever pay for: a black-box solver feeds the Φ₁ dial; healthy readings become a certified microsecond surrogate cache, walls and out-of-range queries get an honest refusal](https://raw.githubusercontent.com/dimaq12/lastsolve/main/assets/banner.jpg)

**The last few solves you'll ever pay for.**

`lastsolve` is a certified accelerator and identification layer over *any* black-box parametric solver. You bring an expensive function `f(k) → field` — a PDE solver, a legacy code behind a subprocess, a simulator you can't even import. `lastsolve` spends a handful of calls on Chebyshev nodes and hands back the whole price list:

> **In plain terms.** You have a simulation that takes 2 minutes, and your workflow — a parameter sweep, a calibration loop, an MCMC — needs to call it 10,000 times. `lastsolve` runs it ~10 times, learns how the answer depends on the parameter, and serves the other 9,990 calls in microseconds — **with a validated error bar on every answer, and a typed exception instead of a guess whenever it can't keep that promise.**

**The contract:** `f(k) → 1-D numpy array` (anything `np.asarray`-able; a scalar result is a length-1 array). Plain CPU NumPy in, plain NumPy out — no PyTorch/JAX tensors needed, no GPU, no training loops. The whole library is a few files of numpy + scipy + [resona](https://pypi.org/project/resona/) you can read in an evening.

**Learn by doing:** the **[COOKBOOK](COOKBOOK.md)** — ten paste-and-run recipes, each ending with the verbatim output it printed on this machine.

```python
from lastsolve import accelerate, learn

@accelerate(warmup=5)        # watch 5 real calls, learn the range they live in,
def solve(k):                # then serve everything from the surrogate
    ...  # your expensive solver

solve(0.021)                 # microseconds, with a validated error and the Φ₁ dial
solve.stats                  # every real call counted

s = learn(solve_fn, (0.014, 0.026))          # one knob  → Surrogate
s = learn(solve_fn, [(a1,b1), (a2,b2)])      # several   → SurrogateND, same verbs
k_hat, crb = s.invert(y_obs)                 # rich Estimate, unpacks as a tuple
s.certify(n_cal=8)                           # Certificate(err ≤ 3.9e-15, ≥88.9%)
s.query(0.05)                                # OutOfRangeError — lastsolve refuses
                                             # to extrapolate; strict=False to
                                             # accept an uncertified answer knowingly
```

Built on [`resona`](https://pypi.org/project/resona/)'s matrix-free effective-rank dial — the measuring instrument of the *Spectra Without Matrices* series. The philosophy in one line: **measure the structure first, then pay accordingly.**

## What you get for ~10 solves

| API | What it does | The honest part |
|---|---|---|
| `@accelerate` | transparent surrogate cache: in-range calls served in µs at near machine precision | out-of-range calls fall through to the real solver — never extrapolates; `.stats` counts every real call |
| `Surrogate(f, krange)` | the core: `query(k)`, `deriv(k)` (Fisher info), adaptive node ladder, `transform='auto'` discovers coordinates like 1/√k | `.val_err` from held-out solves; `.phi1` — resona's dial: ~1 healthy, ≫1 a wall, ~0 dead parameter |
| `.certify(n_cal, alpha)` | split-conformal error band | **distribution-free finite-sample guarantee** (and it tells you 8 calibration points buy 88.9%, not 90%) |
| `identify(f, data, krange)` | maximum-likelihood k̂ ± Cramér–Rao bar from one observation | verdict includes *"the data do not contain this parameter"* when they don't |
| `audit(f, x0, sigma, prior)` | field-level identifiability: how many independent numbers about N unknowns your dataset holds (matrix-free probes + resona) | reports the **blind** subspace no method can recover, before anyone reconstructs anything |
| `detect_break(f, krange)` | bifurcation alarm + blind localization via validation-error bisection | refuses to fit across broken physics instead of interpolating a lie |
| `classify_wall(f, krange)` | after the alarm: is the wall *removable* (a coordinate heals it) or *genuine* (Shor-class)? | resona's lift-rank saturation test — the Journey-I instrument, pointed at your family |
| `SurrogateND(f, box)` | several parameters, committee-disagreement adaptive design | 45 solves matched a 125-solve tensor grid on a 3-parameter family; sloppiness (Φ₁ ≈ 1–2 at p=3) is measured, not assumed |
| `CommandSolver` + `accelerate_command` | accelerate a solver you cannot import — any CLI/binary | every subprocess invocation counted |
| `TimePropagator(snapshots)` | learn the propagator from one trajectory (DMD/Koopman via resona.lift) | ships \|λ\|max stability verdict + held-out-tail validation; says when long prediction is dishonest |
| `identify_spectral(family, k0, λs)` | recover a parameter from a measured spectral fingerprint (resona.wkernel + rayleigh_polish) | uses several eigenvalues — one alone can be ambiguous, and the docstring says so |
| `normality_warning(A)` | non-normality check before trusting any spectral dial | "the spectrum lies about where this operator acts" is a measurable condition |

## Why believe any of this

Because none of it is asserted — it was all *run first*. The methods were battle-tested on a zoo of 35 nonlinear PDE families (Burgers, KdV, Kuramoto–Sivashinsky chaos, NLS solitons, Camassa–Holm, fractional heat…), included here as `lastsolve.zoo` and exercised by the test suite:

- forward surrogates at ~5·10⁻¹⁵ over ±28% parameter ranges, ~570× faster queries;
- inversion saturating the Cramér–Rao bound (median error ≈ 0.7× the bound — the theoretical optimum is 0.674);
- conformal coverage measured at 88.4% against the exact 88.9% finite-sample guarantee;
- a pitchfork bifurcation localized blind to 3% of the range, zero false alarms on healthy physics.

The research trail with every number: [The Price of an Answer](https://github.com/dimaq12/the-price-of-an-answer) (Journey II) and [Never Quantum at All](https://github.com/dimaq12/do-we-need-quantum-computing) (Journey I).

## Install & test

```bash
pip install lastsolve            # that's it — deps are numpy, scipy, resona:
                                 # no ML frameworks, no CUDA, nothing to train
```

Hacking on it / running the test suite:

```bash
git clone https://github.com/dimaq12/lastsolve.git && cd lastsolve
pip install -e . && pip install pytest
pytest tests/ -q                 # 26 tests, ~17 s, real PDEs inside
```

## Honest limits (v1.1.2)

- Scalar-parameter surrogates are certified; **multi-parameter** (`SurrogateND`) is adaptive least-squares — excellent in the sloppy regime (which is most of physics), but its validation is empirical, not conformal yet.
- A *scalar-k* surrogate is tied to one configuration — but that is not the real limit: **lift the configuration into parameters.** Expand the initial condition (or geometry, or forcing) in a small basis and hand its coefficients to `SurrogateND` alongside `k`; one precompute then covers the whole *family* of ICs, because Φ₁ stays low even in the enlarged space (measured: a 4-D box of viscosity + 3 IC-coefficients hits 2·10⁻⁶ from 120 solves at Φ₁ ≈ 2.5). For *linear* PDEs the transfer is exact via u(T;k) = G(k)·u₀. It is a cache around a *question family* — and the family can be as wide as you are willing to parameterize. See the "recalibrate a whole family" recipe in the [cookbook](COOKBOOK.md).
- Smoothness is measured, not assumed: when the parametric manifold genuinely resists (a breathing soliton — Kolmogorov n-width), Φ₁ flags it and no node budget will help. The dial exists precisely so the library can refuse honestly.
- `resona.effective_rank` is a stochastic (Hutchinson) estimate; `lastsolve` uses 128 probes, good to a few percent on the dial.

## Design principles

1. **Never silently wrong** — every answer is validated, certified, or refused.
2. **The dial before the fit** — Φ₁ says whether a cheap surrogate exists at all.
3. **Everything is counted** — the solver calls we spend are the price we quote.

---
*MIT · Dmytro Sierikov · part of the Spectra Without Matrices series*
