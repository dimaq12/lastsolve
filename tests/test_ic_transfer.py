"""The README claim, made a test: a fixed-IC surrogate does NOT transfer to a
new initial condition, but lifting the IC into the parameter vector DOES —
one precompute covers the whole family. Both halves are asserted, so neither
the limitation nor the escape hatch can silently regress."""
import numpy as np

from lastsolve import Surrogate, learn
from lastsolve.zoo import X, K2, strang, d_dx


def _burgers(k, c):
    """Burgers at viscosity k, IC = c·{sin x, sin2x, sin3x}."""
    u0 = c[0]*np.sin(X) + c[1]*np.sin(2*X) + c[2]*np.sin(3*X)
    return np.real(strang(u0, 0.12, 64, k*K2, lambda u: -u*d_dx(u)))


def _median_err(surrogate_query, truth_fn, points, rng):
    errs = []
    for p in points:
        t = truth_fn(p)
        errs.append(np.linalg.norm(surrogate_query(p)-t)/np.linalg.norm(t))
    return float(np.median(errs))


def test_fixed_ic_surrogate_does_not_transfer():
    """Half 1: a k-only surrogate is genuinely tied to its IC."""
    c_ref = np.array([1.0, 0.3, 0.0])
    s = Surrogate(lambda k: _burgers(k, c_ref), (0.014, 0.026)).fit()
    rng = np.random.default_rng(0)
    ks = [0.014 + 0.012*rng.random() for _ in range(20)]

    same = _median_err(lambda k: s.query(k), lambda k: _burgers(k, c_ref), ks, rng)
    assert same < 1e-10                          # on its own IC: machine precision

    c_new = np.array([0.9, 0.35, 0.05])
    moved = _median_err(lambda k: s.query(k), lambda k: _burgers(k, c_new), ks, rng)
    assert moved > 1e-2                          # a different IC: badly wrong


def test_lifting_ic_into_parameters_covers_the_family():
    """Half 2: put the IC coefficients in the parameter vector and one
    precompute covers the whole (k, IC) family — the README's escape hatch."""
    box = [(0.014, 0.026), (0.7, 1.3), (0.1, 0.5), (-0.1, 0.2)]
    f = lambda th: _burgers(th[0], th[1:])
    s = learn(f, box, budget=120)
    assert s.solves <= 120
    assert s.phi1 < 4                            # sloppiness survives the extra axes

    rng = np.random.default_rng(7)
    pts = [np.array([lo + (hi-lo)*rng.random() for lo, hi in box])
           for _ in range(40)]
    med = _median_err(lambda th: s.query(th), f, pts, rng)
    assert med < 1e-4                            # whole family, one precompute
