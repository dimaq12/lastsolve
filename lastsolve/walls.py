"""
walls.py — refuse to interpolate across broken physics.

A surrogate fitted across a bifurcation returns confident nonsense. The
detector here uses the instrument lastsolve already trusts — a Chebyshev
kernel's held-out validation error: smooth family ⇒ ~1e-12, a kink inside
the range ⇒ ~1e-2, orders of magnitude apart. On alarm, bisection into the
SICKER half localizes the break blind; the caller can then split the range
(with a margin — near the critical point the physics itself is singular)
and refit each branch, e.g. in the coordinate p = √(k − k̂) that straightens
a pitchfork.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surrogate import Surrogate


@dataclass
class WallReport:
    alarmed: bool
    k_hat: float | None
    val_err_full: float
    solves: int

    def __repr__(self):
        if not self.alarmed:
            return f"WallReport(healthy, val_err={self.val_err_full:.1e})"
        return (f"WallReport(WALL at k̂≈{self.k_hat:.4g}, "
                f"val_err={self.val_err_full:.1e}, solves={self.solves})")


def detect_break(forward, krange, bad=1e-6, levels=6, seed=0):
    """Alarm + blind localization of a qualitative change inside krange."""
    ka, kb = float(krange[0]), float(krange[1])
    solves = 0

    def fit(lo, hi, ladder, sd):
        s = Surrogate(forward, (lo, hi))
        s.fit(ladder=ladder, seed=sd)
        return s

    full = fit(ka, kb, (15,), seed+10)
    solves += full.solves
    if full.val_err < bad:
        return WallReport(False, None, full.val_err, solves)
    lo, hi = ka, kb
    for lvl in range(levels):
        mid = 0.5*(lo+hi)
        left = fit(lo, mid, (7,), seed+20+lvl)
        right = fit(mid, hi, (7,), seed+40+lvl)
        solves += left.solves + right.solves
        bad_l, bad_r = left.val_err > bad, right.val_err > bad
        if not bad_l and not bad_r:
            return WallReport(True, mid, full.val_err, solves)
        if left.val_err >= right.val_err:
            hi = mid
        else:
            lo = mid
    return WallReport(True, 0.5*(lo+hi), full.val_err, solves)
