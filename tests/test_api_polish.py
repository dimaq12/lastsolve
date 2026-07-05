"""Diamond-cut API tests: learn(), strict ranges, typed refusals, rich
results, warmup mode — the ergonomics ARE the honesty, so they get tests."""
import numpy as np
import pytest

from lastsolve import (learn, accelerate, Surrogate, SurrogateND,
                       Estimate, Certificate, NotFittedError, OutOfRangeError)


def cheap(k):                       # analytic stand-in for an expensive solver
    x = np.linspace(0, 2*np.pi, 48, endpoint=False)
    return np.sin(x)*np.exp(-3*k) + k*np.cos(2*x)


def cheap3(k):
    x = np.linspace(0, 2*np.pi, 48, endpoint=False)
    return k[0]*np.sin(x) + k[1]*np.cos(2*x) + k[2]*np.sin(3*x)**2


def test_learn_dispatches_scalar_and_nd():
    s = learn(cheap, (0.5, 1.5))
    assert isinstance(s, Surrogate) and s.val_err < 1e-10
    snd = learn(cheap3, [(0.5, 1.5), (0.2, 0.8), (1.0, 2.0)], budget=30)
    assert isinstance(snd, SurrogateND) and snd.val_err < 1e-8


def test_strict_range_refusal_and_conscious_override():
    s = learn(cheap, (0.5, 1.5))
    with pytest.raises(OutOfRangeError):
        s.query(2.0)
    y = s.query(2.0, strict=False)          # knowingly uncertified
    assert np.all(np.isfinite(y))


def test_not_fitted_refusal_is_typed_and_helpful():
    s = Surrogate(cheap, (0.5, 1.5))
    with pytest.raises(NotFittedError, match="fit"):
        s.query(1.0)


def test_estimate_is_rich_and_unpacks():
    s = learn(cheap, (0.5, 1.5))
    est = s.invert(cheap(1.1))
    k_hat, crb = est                        # tuple protocol
    assert isinstance(est, Estimate)
    assert abs(k_hat-1.1) < 1e-6
    assert "±" in repr(est) and est.verdict


def test_certificate_is_typed_and_stored():
    s = learn(cheap, (0.5, 1.5))
    cert = s.certify(n_cal=8, alpha=0.1)
    assert isinstance(cert, Certificate)
    assert cert.guarantee == pytest.approx(8/9)
    assert s.certificate is cert
    assert "coverage" in repr(cert) and repr(s).count("Φ₁") == 1


def test_accelerate_warmup_builds_range_from_observed_calls():
    calls = {"n": 0}

    def counted(k):
        calls["n"] += 1
        return cheap(k)

    fast = accelerate(counted, warmup=5)
    ks = [0.8, 1.3, 0.95, 1.1, 1.02]
    for k in ks:                            # warmup: all real
        fast(k)
    assert calls["n"] >= 5
    built = calls["n"]
    lo, hi = fast.surrogate.ka, fast.surrogate.kb
    assert lo < min(ks) and hi > max(ks)    # range covers what was seen
    for k in ks:
        fast(k)                             # now served
    assert calls["n"] == built
    assert "served" in repr(fast)


def test_nd_estimate_verdict_and_edge_jacobian():
    snd = learn(cheap3, [(0.5, 1.5), (0.2, 0.8), (1.0, 2.0)], budget=30)
    est = snd.invert(cheap3(np.array([1.4999, 0.5, 1.5])))
    k_hat, crb = est
    assert isinstance(est, Estimate) and est.verdict
    assert np.all(np.abs(k_hat-[1.4999, 0.5, 1.5]) < 1e-3)  # edge: jac must not raise
