"""
identify.py — parameter identification with an honest error bar, one call.

    result = lastsolve.identify(solve, data=u_obs, krange=(0.7*k0, 1.3*k0))
    result.k_hat, result.crb        # k̂ ± Cramér–Rao bar
    result.verdict                  # human-readable honesty

The estimator is maximum likelihood on a Chebyshev surrogate of the forward
map (optionally polished on the true solver), and the bar is σ̂/‖∂f/∂k‖ with
the noise level σ̂ estimated from the data themselves. When the sensitivity
is too weak, the verdict says "the data do not contain this parameter" —
instead of returning a confidently wrong number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surrogate import Surrogate


@dataclass
class IdentifyResult:
    k_hat: float
    crb: float
    phi1: float
    solves: int
    verdict: str

    def __repr__(self):
        return (f"IdentifyResult(k_hat={self.k_hat:.6g}, ±{self.crb:.2g} (CRB), "
                f"phi1={self.phi1:.2f}, solves={self.solves}, '{self.verdict}')")


def identify(forward, data, krange, polish=True, transform=None,
             surrogate=None, seed=0):
    """Recover the parameter that produced `data` from a black-box forward map.

    Pass a prefit `surrogate` to pay ZERO additional solver calls.
    """
    data = np.asarray(data, dtype=float)
    s = surrogate
    if s is None:
        s = Surrogate(forward, krange, transform=transform)
        s.fit(seed=seed)
    k_hat, crb = s.invert(data, polish=polish)
    rng_width = s.kb - s.ka
    if not np.isfinite(crb) or crb > 0.5*rng_width:
        verdict = ("the data do not contain this parameter "
                   f"(CRB {crb:.2g} vs range {rng_width:.2g})")
    elif crb > 0.05*rng_width:
        verdict = "weakly identifiable — treat k_hat as an interval, not a number"
    else:
        verdict = "identifiable; error bar is the Cramér–Rao floor"
    return IdentifyResult(k_hat=float(k_hat), crb=float(crb),
                          phi1=s.phi1, solves=s.solves, verdict=verdict)
