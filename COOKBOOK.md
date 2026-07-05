# lastsolve — the cookbook

Task → verb → snippet → the lines it prints. Every output block below is the
verbatim result of running the snippet on this machine — not a target, a
transcript. `pip install -e .` first (or set `PYTHONPATH` to the repo), then
paste any recipe into a file and run it.

The contract everywhere: your solver is a plain function `f(k) → 1-D
numpy array` (anything `np.asarray`-able; a scalar output is just a length-1
array). No PyTorch, no CUDA, no training loops — numpy + scipy + resona.

---

## ACCELERATE

### A 2000-point sweep for the price of 25 solves

Your simulation takes minutes and your workflow calls it thousands of times.
One decorator: the first `warmup` calls run for real, lastsolve learns the
range they live in, and everything after is served from a validated surrogate.

```python
import numpy as np
from lastsolve import accelerate

calls = {"n": 0}
def slow_sim(k):                      # pretend each call takes minutes
    calls["n"] += 1
    x = np.linspace(0, 2*np.pi, 64, endpoint=False)
    return np.sin(x)*np.exp(-3*k) + k*np.cos(2*x)

fast = accelerate(slow_sim, warmup=5)
for k in (0.8, 1.3, 0.95, 1.1, 1.02):          # your workflow, unchanged
    fast(k)
sweep = [fast(k) for k in np.linspace(0.85, 1.25, 2000)]
print(fast)
print("real solver calls for a 2000-point sweep:", calls["n"])
```
```
Accelerated(25 real → 2001 served, 0 fallbacks; range 0.725..1.375, val_err 7.5e-13, Φ₁ 1.20)
real solver calls for a 2000-point sweep: 25
```

The receipt is the point: 25 real calls, 2001 served, validated error 7.5e-13,
and a dial reading Φ₁ = 1.2 (healthy). A call outside 0.725..1.375 would fall
through to the real solver and be counted — never extrapolated.

### A solver you cannot even import (CLI / binary / Fortran relic)

```python
import sys, os, tempfile
from lastsolve import CommandSolver, accelerate_command

code = ("import sys, numpy as np\n"
        "k = float(sys.argv[1])\n"
        "x = np.linspace(0, 6.283185307179586, 64, endpoint=False)\n"
        "u = np.sin(x)*np.exp(-25*k) + k*np.cos(2*x)\n"
        "print(' '.join('%.17g' % v for v in u))\n")
path = os.path.join(tempfile.gettempdir(), "legacy_solver.py")
open(path, "w").write(code)
legacy = CommandSolver(sys.executable + " " + path + " {k}")   # a black box
s = accelerate_command(legacy, krange=(0.014, 0.026))
print(s)
print("subprocess invocations:", legacy.calls)
```
```
Surrogate([0.014, 0.026] in 'identity' coords: 7 nodes, 10 solves, val_err 5.4e-12, Φ₁ 0.98)
subprocess invocations: 10
```

Ten launches of a process we never imported buy microsecond answers over the
whole range. Swap the template for `"./mysolver --visc {k}"` and a `parser=`
for your output format.

---

## CERTIFY

### Turn "it seems accurate" into a distribution-free guarantee

```python
import numpy as np
from lastsolve import learn
from lastsolve.zoo import zoo, make_observable

pde = next(p for p in zoo() if p.name == "Burgers")
obs = make_observable(pde)
f = lambda k: obs(pde.u0, pde.t, [k])
s = learn(f, (0.7*pde.k0[0], 1.3*pde.k0[0]))
print(s)
print(s.certify(n_cal=8, alpha=0.1))
```
```
Surrogate([0.014, 0.026] in 'identity' coords: 7 nodes, 10 solves, val_err 4.9e-15, Φ₁ 1.20)
Certificate(err ≤ 8.25e-15 with ≥88.9% coverage; 8 calibration solves)
```

Split conformal prediction: 8 fresh calibration solves buy the guarantee
"a fresh random in-range query errs ≤ 8.25e-15 with probability ≥ 8/9" — by
exchangeability alone, no smoothness assumptions. Note the honest 88.9%:
that is the exact finite-sample value; a literal 90% at n=8 does not exist.

---

## IDENTIFY

### A hidden parameter, with the error bar physics allows — and an honest "no"

```python
import numpy as np
from lastsolve import identify
from lastsolve.zoo import zoo, make_observable

pde = next(p for p in zoo() if p.name == "Burgers")
obs = make_observable(pde)
f = lambda k: obs(pde.u0, pde.t, [k])
rng = np.random.default_rng(11)
y = f(1.1*pde.k0[0])                                   # hidden truth: 0.022
print(identify(f, y + rng.normal(0, 2e-4*np.max(np.abs(y)), y.size),
               (0.7*pde.k0[0], 1.3*pde.k0[0]), polish=False))
print(identify(f, y + rng.normal(0, 5e-2*np.max(np.abs(y)), y.size),
               (0.7*pde.k0[0], 1.3*pde.k0[0]), polish=False))
```
```
IdentifyResult(k̂ = 0.0221511 ± 0.00013 (CRB), Φ₁ 1.20, 10 solves, 'identifiable; error bar is the Cramér–Rao floor')
IdentifyResult(k̂ = 0.014 ± 0.032 (CRB), Φ₁ 1.20, 10 solves, 'the data do not contain this parameter (CRB 0.032 vs range 0.012)')
```

Same solver, same code — two verdicts. At low noise: k̂ within 1.2σ of the
truth, and the bar is the Cramér–Rao floor (no estimator can beat it). At 5%
noise the honest answer is that the answer is not in the data — and lastsolve
says exactly that instead of returning a confident number.

### A full Bayesian posterior for ten solver calls

```python
import numpy as np
from lastsolve import learn
from lastsolve.zoo import zoo, make_observable

pde = next(p for p in zoo() if p.name == "Burgers")
obs = make_observable(pde)
f = lambda k: obs(pde.u0, pde.t, [k])
ka, kb = 0.7*pde.k0[0], 1.3*pde.k0[0]
s = learn(f, (ka, kb))
rng = np.random.default_rng(3)
data = f(0.9*pde.k0[0]); sigma = 0.02*np.max(np.abs(data))
data = data + rng.normal(0, sigma, data.size)
grid = np.linspace(ka, kb, 2001)                       # 2001 likelihood evals…
logp = np.array([-np.sum((s.query(k)-data)**2)/(2*sigma**2) for k in grid])
w = np.exp(logp-logp.max()); w /= w.sum()
mean = float((grid*w).sum()); std = float(np.sqrt(((grid-mean)**2*w).sum()))
print(f"posterior: k = {mean:.6f} ± {std:.6f}   (solver calls: {s.solves})")
```
```
posterior: k = 0.019812 ± 0.003420   (solver calls: 10)
```

2001 likelihood evaluations, 10 solver calls: the truth (0.018) sits at 0.5
posterior-σ. The research stand behind this recipe matched a 400-true-solve
reference posterior to total-variation 5·10⁻¹³. MCMC over `s.query` is just
as free.

### Several knobs at once

```python
import numpy as np
from lastsolve import learn
from lastsolve.zoo import X, K2, strang, d_dx, smooth

def f3(k):        # Burgers with 3 knobs: viscosity, advection, forcing
    return np.real(strang(smooth, 0.15, 64, k[0]*K2,
                          lambda u: -k[1]*u*d_dx(u) + k[2]*np.sin(X)))

s = learn(f3, [(0.016, 0.024), (0.8, 1.2), (0.24, 0.36)], budget=45)
print(s)
print(s.invert(f3(np.array([0.021, 0.93, 0.30]))))
```
```
SurrogateND(p=3, 42 training solves (45 total), val_err 2.1e-07, Φ₁ 1.51)
Estimate(k̂ = [0.021, 0.93, 0.3] ± [1e-08, 2e-08, 2e-08] (CRB), 'identifiable; bars are the Cramér–Rao floor')
```

Three parameters, 45 solves, all three recovered with per-parameter bars.
Note Φ₁ = 1.51 at p = 3: physics is *sloppy* — the response occupies far
fewer directions than the knob count, which is exactly why so few solves
suffice.

### From a spectral fingerprint (you measured frequencies, not fields)

```python
import numpy as np
from lastsolve import identify_spectral

T0 = np.diag(2.0*np.ones(24)) - np.diag(np.ones(23), 1) - np.diag(np.ones(23), -1)
D = np.diag(np.linspace(0.5, 1.5, 24))
family = lambda k: k*T0 + k*k*D
lam_obs = np.linalg.eigvalsh(family(0.83))[:4]        # measured fingerprint
k_hat, lam_hat = identify_spectral(family, k0=1.0, lam_targets=lam_obs)
print(f"k̂ = {k_hat:.12f}   (truth 0.83)   max |Δλ| = {np.max(np.abs(lam_hat-lam_obs)):.1e}")
```
```
k̂ = 0.830000000000   (truth 0.83)   max |Δλ| = 5.6e-16
```

Hellmann–Feynman sensitivities (`resona.wkernel`) + a design step + Rayleigh
polish: the parameter to twelve digits from four eigenvalues. Several
eigenvalues, deliberately — a single one can be matched by more than one k.

---

## AUDIT & REFUSE

### Before reconstructing anything: how much answer do the data contain?

```python
import numpy as np
from lastsolve import audit

rng = np.random.default_rng(0)
A = rng.standard_normal((64, 5)) @ rng.standard_normal((5, 32))
forward = lambda x: A @ x            # 32 unknowns observed through rank-5 physics
print(audit(forward, np.zeros(32), sigma=1e-6, prior_amp=1.0,
            n_probes=12, seed=1))
```
```
AuditReport(phi1=3.59, visible≈5 of 12 probed directions, solves=24, 'the data contain ≈5 independent numbers about 32 unknowns; the rest is blind for any method')
```

24 matrix-free probes find the true rank (5) of what the observation can see.
The other 27 directions of the unknown are blind — for any method, including
one published next year. Reconstruct the five, refuse the rest.

### When the physics breaks inside your range

```python
import numpy as np
from lastsolve import detect_break, classify_wall
from lastsolve.zoo import X, K2, strang

def ac(k):        # Allen–Cahn: pitchfork bifurcation at k ≈ 0.001
    return np.real(strang(0.1*np.sin(X), 40.0, 600, 0.001*K2 - k,
                          lambda u: -u**3))

print(detect_break(ac, (-0.3, 0.5)))
print(classify_wall(ac, (-0.3, 0.5), n_samples=160, windows=(16, 32, 48, 64)))
```
```
WallReport(WALL at k̂≈-0.025, val_err=2.0e+00, solves=118)
WallClass('removable', lift-ranks [1.0, 1.02, 1.48, 1.87], solves=160)
```

The detector never sees the true transition point; it localizes it blind to
3% of the range. The classifier (resona's lift-rank saturation — the same
test that separated Shor's wall from a periodic signal in Journey I) says
*removable*: a coordinate heals this wall — split at k̂ and refit the upper
branch in `transform=(lambda k: np.sqrt(k-k_hat), lambda p: k_hat+p**2)`.
A *genuine* verdict would mean: stop spending, no chart exists.

---

## DYNAMICS

### Learn the propagator from one trajectory — with a stability receipt

```python
import numpy as np
from lastsolve import TimePropagator
from lastsolve.zoo import K2, strang, smooth

u, snaps = smooth.copy(), [smooth.copy()]
for _ in range(60):
    u = np.real(strang(u, 0.02, 4, 0.05*K2, lambda w: 0*w))
    snaps.append(u.copy())
print(TimePropagator(np.array(snaps)))
```
```
TimePropagator(rank=2, |λ|max=0.9990 [stable], val_err=1.6e-14 over 12 held-out steps)
```

DMD/Koopman via `resona.lift.koopman`: the trajectory's own tail is held out
and predicted to 1.6e-14. If the learned modes had |λ|max > 1 the repr would
say `[UNSTABLE — long prediction is dishonest]` — the refusal is part of the
object.

---

## Reading the dial

| Φ₁ reading | Meaning | What lastsolve does |
|---|---|---|
| ≈ 1 | one-dimensional response manifold — the cheap regime | machine-precision surrogate, certified |
| 1–3 | few directions (sloppy multi-parameter physics) | adaptive designs still win big |
| ≫ 1 (e.g. 2.7+) | genuinely curved manifold — an n-width wall (breathing soliton class) | flags it; no node budget will help, and it says so |
| ≈ 0 | the parameter barely moves the observable | identification refused; try a spectral observable |
