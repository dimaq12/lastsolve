"""End-to-end tests: every public API on real PDEs from the zoo fixture."""
import numpy as np
import pytest

from lastsolve import (Surrogate, accelerate, identify, audit, detect_break,
                       classify_wall, hard_points)
from lastsolve.zoo import zoo, make_observable, X, K2, strang


def burgers_forward():
    pde = next(p for p in zoo() if p.name == "Burgers")
    obs = make_observable(pde)
    return pde, (lambda k: obs(pde.u0, pde.t, [k]))


# ── Surrogate: forward, dial, inverse, certificate ───────────────────────────
def test_surrogate_forward_machine_precision():
    pde, f = burgers_forward()
    s = Surrogate(f, (0.7*pde.k0[0], 1.3*pde.k0[0])).fit()
    assert s.val_err < 1e-10
    rng = np.random.default_rng(3)
    for _ in range(5):
        k = pde.k0[0]*(0.72+0.56*rng.random())
        y = f(k)
        err = np.linalg.norm(s.query(k)-y)/np.linalg.norm(y)
        assert err < 1e-9


def test_surrogate_phi1_healthy():
    pde, f = burgers_forward()
    s = Surrogate(f, (0.7*pde.k0[0], 1.3*pde.k0[0])).fit()
    assert 0.8 < s.phi1 < 1.5          # healthy one-dimensional manifold


def test_surrogate_invert_clean():
    pde, f = burgers_forward()
    s = Surrogate(f, (0.7*pde.k0[0], 1.3*pde.k0[0])).fit()
    k_true = 0.93*pde.k0[0]
    k_hat, crb = s.invert(f(k_true), polish=True)
    assert abs(k_hat-k_true) < 1e-8


def test_surrogate_certify():
    pde, f = burgers_forward()
    s = Surrogate(f, (0.7*pde.k0[0], 1.3*pde.k0[0])).fit()
    cert = s.certify(n_cal=8, alpha=0.1)
    assert cert["guarantee"] == pytest.approx(8/9)
    assert cert["band"] < 1e-9         # smooth family: tiny certified band


# ── @accelerate: counting and honesty ────────────────────────────────────────
def test_accelerate_counts_and_accuracy():
    pde, f = burgers_forward()
    calls = {"n": 0}

    def counted(k):
        calls["n"] += 1
        return f(k)

    fast = accelerate(counted, span=0.3)
    k0 = pde.k0[0]
    first = fast(k0)                    # triggers the eager build
    built = calls["n"]
    assert built <= 14                  # 7-11 nodes + 3 validation
    rng = np.random.default_rng(5)
    for _ in range(50):
        k = k0*(0.75+0.5*rng.random())
        y_fast = fast(k)
        assert calls["n"] == built      # served, not solved
        err = np.linalg.norm(y_fast-f(k))/np.linalg.norm(f(k))
        assert err < 1e-9
    fast(2.0*k0)                        # out of range → honest fallback
    assert fast.stats["fallbacks_out_of_range"] == 1
    assert calls["n"] == built + 1


# ── identify(): honest error bar ─────────────────────────────────────────────
def test_identify_noisy_within_bars():
    pde, f = burgers_forward()
    k_true = 1.1*pde.k0[0]
    y = f(k_true)
    rng = np.random.default_rng(11)
    # low noise: parameter identifiable, k̂ lands within the stated bar
    noisy = y + rng.normal(0, 2e-4*np.max(np.abs(y)), y.size)
    res = identify(f, noisy, (0.7*pde.k0[0], 1.3*pde.k0[0]), polish=False)
    assert np.isfinite(res.crb)
    assert abs(res.k_hat-k_true) < 5*res.crb
    assert res.verdict.startswith("identifiable")


def test_identify_honest_refusal_at_high_noise():
    # at 5% noise Burgers viscosity is NOT identifiable on this horizon —
    # the verdict must say so instead of returning a confident number
    pde, f = burgers_forward()
    y = f(1.1*pde.k0[0])
    rng = np.random.default_rng(12)
    noisy = y + rng.normal(0, 0.05*np.max(np.abs(y)), y.size)
    res = identify(f, noisy, (0.7*pde.k0[0], 1.3*pde.k0[0]), polish=False)
    assert ("do not contain" in res.verdict) or ("weakly" in res.verdict)


# ── audit(): rank of what the data contain ───────────────────────────────────
def test_audit_linear_rank5():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((64, 5)) @ rng.standard_normal((5, 32))
    forward = lambda x: A @ x
    rep = audit(forward, np.zeros(32), sigma=1e-6, prior_amp=1.0,
                n_probes=12, seed=1)
    assert 3 <= rep.visible <= 6        # sketch sees ≈ the true rank 5
    assert rep.phi1 < 6


# ── detect_break(): wall alarm, healthy silence ──────────────────────────────
def _allen_cahn(k):
    return np.real(strang(0.1*np.sin(X), 40.0, 600, 0.001*K2 - k,
                          lambda u: -u**3))


def test_wall_alarm_and_blind_localization():
    rep = detect_break(_allen_cahn, (-0.3, 0.5))
    assert rep.alarmed
    assert abs(rep.k_hat-0.001) < 0.05*0.8      # within 5% of the range


def test_wall_silent_on_healthy():
    pde, f = burgers_forward()
    rep = detect_break(f, (0.65*pde.k0[0], 1.35*pde.k0[0]))
    assert not rep.alarmed


def test_classify_wall_pitchfork_is_removable():
    # the pitchfork is a shock-type singularity: a finite lift linearizes it
    # (we heal it with √(k−k̂)) — resona's lift-rank test must agree
    rep = classify_wall(_allen_cahn, (-0.3, 0.5), n_samples=160,
                        windows=(16, 32, 48, 64))
    assert rep.kind == 'removable'
    assert rep.solves == 160


def test_sensitivity_richardson_beats_plain_fd():
    pde, f = burgers_forward()
    s = Surrogate(f, (0.7*pde.k0[0], 1.3*pde.k0[0])).fit()
    k = 1.05*pde.k0[0]
    truth = s.deriv(k)                       # interpolant derivative ~1e-14
    eps = 0.02*pde.k0[0]                     # deliberately coarse step
    plain = s.sensitivity(k, refine=False, eps=eps)
    refined = s.sensitivity(k, refine=True, eps=eps)
    e_plain = np.linalg.norm(plain-truth)/np.linalg.norm(truth)
    e_ref = np.linalg.norm(refined-truth)/np.linalg.norm(truth)
    assert e_ref < e_plain/10


def test_hard_points_avoided_crossing():
    fam = lambda k: np.array([[k, 0.05], [0.05, -k]])
    ks = np.linspace(-0.5, 0.5, 41)
    k_star, profile = hard_points(fam, ks, np.eye(2))
    assert abs(k_star) <= 0.025              # gap minimum is at k = 0
    assert len(profile) == len(ks)
