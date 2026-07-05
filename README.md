# lastsolve

**The last few solves you'll ever pay for.**

`lastsolve` is a certified accelerator and identification layer over *any* black-box parametric solver. You bring an expensive function `f(k) → field` — a PDE solver, a legacy code behind a subprocess, a simulator you can't even import. `lastsolve` spends a handful of calls on Chebyshev nodes and hands back the whole price list:

```python
from lastsolve import accelerate

@accelerate
def solve(k):
    ...  # your expensive solver

solve(0.021)     # ~10 real solves once — then microseconds,
solve.stats      # with a validated error and the Φ₁ dial
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

## Why believe any of this

Because none of it is asserted — it was all *run first*. The methods were battle-tested on a zoo of 35 nonlinear PDE families (Burgers, KdV, Kuramoto–Sivashinsky chaos, NLS solitons, Camassa–Holm, fractional heat…), included here as `lastsolve.zoo` and exercised by the test suite:

- forward surrogates at ~5·10⁻¹⁵ over ±28% parameter ranges, ~570× faster queries;
- inversion saturating the Cramér–Rao bound (median error ≈ 0.7× the bound — the theoretical optimum is 0.674);
- conformal coverage measured at 88.4% against the exact 88.9% finite-sample guarantee;
- a pitchfork bifurcation localized blind to 3% of the range, zero false alarms on healthy physics.

The research trail with every number: [The Price of an Answer](https://github.com/dimaq12/the-price-of-an-answer) (Journey II) and [Never Quantum at All](https://github.com/dimaq12/do-we-need-quantum-computing) (Journey I).

## Install & test

```bash
pip install numpy scipy resona
pip install -e .
pytest tests/ -q        # 10 tests, ~6 s, real PDEs inside
```

## Honest limits (v0.1)

- **Single scalar parameter.** Multi-parameter adaptive designs are prototyped (45 solves matched a 125-solve tensor grid on a 3-parameter family) and land in v0.2.
- The surrogate is built **per initial condition / configuration** — it is a cache around a question family, not a general solver replacement.
- Smoothness is measured, not assumed: when the parametric manifold genuinely resists (a breathing soliton — Kolmogorov n-width), Φ₁ flags it and no node budget will help. The dial exists precisely so the library can refuse honestly.
- `resona.effective_rank` is a stochastic (Hutchinson) estimate; `lastsolve` uses 128 probes, good to a few percent on the dial.

## Design principles

1. **Never silently wrong** — every answer is validated, certified, or refused.
2. **The dial before the fit** — Φ₁ says whether a cheap surrogate exists at all.
3. **Everything is counted** — the solver calls we spend are the price we quote.

---
*MIT · Dmytro Sierikov · part of the Spectra Without Matrices series*
