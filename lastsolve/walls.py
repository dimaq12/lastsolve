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
import resona.cost
from resona.defect import hard_points as hard_points  # re-export: for OPERATOR
# families H(k), resona's response-susceptibility scan finds avoided crossings /
# transitions matrix-free — use it when your problem exposes an operator, not
# just a field map: k_star, profile = hard_points(family, ks, B).

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


@dataclass
class WallClass:
    kind: str                 # 'removable' | 'genuine'
    ranks: list
    solves: int

    def __repr__(self):
        return (f"WallClass('{self.kind}', lift-ranks {self.ranks}, "
                f"solves={self.solves})")


def classify_wall(forward, krange, n_samples=192, windows=(16, 32, 64, 96),
                  seed=0):
    """After detect_break fires: WHAT KIND of wall is it?

    'removable' — the lift rank of the parametric trajectory SATURATES with
    the window: a finite chart linearizes the singularity (a shock, a
    pitchfork — some coordinate like √(k−k̂) heals it; go find it, e.g. with
    Surrogate(transform='auto') per branch).
    'genuine'   — the rank GROWS with the window: no finite chart exists;
    stop spending nodes and treat it as a hard boundary.

    This is resona.cost.is_extractable — the same lift-rank saturation test
    that separated Shor's wall from a periodic signal in Journey I — pointed
    at the parametric response. Costs n_samples real solves (the signal must
    genuinely cross the wall; a surrogate is not to be trusted there).
    """
    ka, kb = float(krange[0]), float(krange[1])
    rng = np.random.default_rng(seed)
    w = None
    sig = []
    for k in np.linspace(ka, kb, n_samples):
        y = np.asarray(forward(float(k)), dtype=float)
        if w is None:
            w = rng.standard_normal(y.size)
            w /= np.linalg.norm(w)
        sig.append(float(w @ y))
    ok, ranks = resona.cost.is_extractable(np.asarray(sig), windows=windows)
    return WallClass(kind='removable' if ok else 'genuine',
                     ranks=[float(x) for x in np.round(np.asarray(ranks, dtype=float), 2)],
                     solves=n_samples)


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
