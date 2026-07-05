"""v1.0 feature tests: multi-parameter, CLI adapter, time propagation,
spectral identification, non-normality warning."""
import os
import sys
import tempfile

import numpy as np

from lastsolve import (SurrogateND, CommandSolver, accelerate_command,
                       TimePropagator, identify_spectral, normality_warning)
from lastsolve.zoo import X, K2, strang, d_dx, smooth


# ── SurrogateND: 3 parameters, 45 solves ─────────────────────────────────────
def burgers3(k):
    return np.real(strang(smooth, 0.15, 64, k[0]*K2,
                          lambda u: -k[1]*u*d_dx(u) + k[2]*np.sin(X)))


def test_surrogatend_beats_taylor_scale():
    box = [(0.016, 0.024), (0.8, 1.2), (0.24, 0.36)]
    s = SurrogateND(burgers3, box).fit(budget=45, seed=0)
    assert s.solves <= 45
    assert s.val_err < 1e-4              # Taylor baseline was ~1.7e-4 median
    rng = np.random.default_rng(9)
    errs = []
    for _ in range(5):
        k = np.array([lo+(hi-lo)*rng.random() for lo, hi in box])
        y = burgers3(k)
        errs.append(np.linalg.norm(s.query(k)-y)/np.linalg.norm(y))
    assert np.median(errs) < 1e-4
    assert s.phi1 < 4                    # sloppiness: p=3 but few directions


def test_surrogatend_invert():
    box = [(0.016, 0.024), (0.8, 1.2), (0.24, 0.36)]
    s = SurrogateND(burgers3, box).fit(budget=45, seed=0)
    k_true = np.array([0.021, 0.93, 0.3])
    k_hat, crb = s.invert(burgers3(k_true))
    assert np.all(np.abs(k_hat-k_true) < np.maximum(5*crb, 5e-3))


# ── CommandSolver: a solver we never import ──────────────────────────────────
def test_command_solver_accelerates():
    script = os.path.join(tempfile.gettempdir(), "toy_cli_solver.py")
    with open(script, "w") as f:
        f.write("import sys, numpy as np\n"
                "k = float(sys.argv[1])\n"
                "x = np.linspace(0, 6.283185307179586, 64, endpoint=False)\n"
                "u = np.sin(x)*np.exp(-25*k) + k*np.cos(2*x)\n"
                "print(' '.join('%.17g' % v for v in u))\n")
    solver = CommandSolver(sys.executable + " " + script + " {k}")
    s = accelerate_command(solver, krange=(0.014, 0.026))
    assert solver.calls <= 14            # nodes + validation only
    y_direct = solver(0.0203)
    err = np.linalg.norm(s.query(0.0203)-y_direct)/np.linalg.norm(y_direct)
    assert err < 1e-10


# ── TimePropagator: linear heat is learned exactly, receipt included ─────────
def test_timepropagator_heat():
    nu = 0.05
    u = smooth.copy()
    snaps = [u.copy()]
    for _ in range(60):
        u = np.real(strang(u, 0.02, 4, nu*K2, lambda w: 0*w))
        snaps.append(u.copy())
    tp = TimePropagator(np.array(snaps))
    assert tp.stable                     # diffusion: |λ| <= 1
    assert tp.val_err < 1e-8             # held-out tail predicted


# ── identify_spectral: recover k from one measured eigenvalue ────────────────
def test_identify_spectral():
    T0 = np.diag(2.0*np.ones(24)) - np.diag(np.ones(23), 1) - np.diag(np.ones(23), -1)
    D = np.diag(np.linspace(0.5, 1.5, 24))
    family = lambda k: k*T0 + k*k*D      # nonlinear in k
    k_true = 0.83
    lam_targets = np.linalg.eigvalsh(family(k_true))[:4]   # fingerprint
    k_hat, lam_hat = identify_spectral(family, k0=1.0, lam_targets=lam_targets)
    assert abs(k_hat-k_true) < 1e-9
    assert np.allclose(lam_hat, lam_targets, atol=1e-9)


# ── normality_warning: symmetric silent, Jordan loud ─────────────────────────
def test_normality_warning():
    S = np.diag([3., 2., 1.])
    rel_s, warn_s = normality_warning(A=S)
    assert warn_s is None and rel_s < 1e-12
    J = np.eye(8, k=1)*10.0 + np.eye(8)*0.1
    rel_j, warn_j = normality_warning(A=J)
    assert warn_j is not None and rel_j > 1e-2
