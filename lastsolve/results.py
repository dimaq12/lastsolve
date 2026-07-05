"""
results.py — the shapes every lastsolve answer comes in, and the errors it
refuses politely with.

Design rules:
  • every estimate is a dataclass with a notebook-grade __repr__ — the most
    important facts are visible without a single print statement;
  • estimates UNPACK like tuples (`k_hat, crb = s.invert(y)`), so the rich
    type costs the caller nothing;
  • refusals are typed: NotFittedError and OutOfRangeError carry the fix in
    their message. Silence is the only thing this library never returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class LastsolveError(Exception):
    """Base class for lastsolve refusals."""


class NotFittedError(LastsolveError):
    """Raised when a surrogate is queried before .fit()."""

    def __init__(self, what="Surrogate"):
        super().__init__(
            f"{what} is not fitted yet — call .fit() first "
            f"(it will spend and count the solver calls it needs).")


class OutOfRangeError(LastsolveError):
    """Raised on a query outside the fitted range: lastsolve does not
    extrapolate silently — that would be a confident lie."""

    def __init__(self, k, lo, hi):
        super().__init__(
            f"k = {k!r} is outside the fitted range [{lo!r}, {hi!r}]. "
            f"lastsolve never extrapolates. Options: call the real solver, "
            f"refit with a wider range, or pass strict=False to accept an "
            f"UNCERTIFIED extrapolation knowingly.")


@dataclass
class Estimate:
    """A parameter estimate with its honesty attached.

    Unpacks as (k_hat, crb):   k_hat, crb = surrogate.invert(y)
    """
    k_hat: object                       # float (scalar) or ndarray (multi)
    crb: object                         # matching Cramér–Rao bar(s)
    verdict: str = ""
    solves: int = 0

    def __iter__(self):
        return iter((self.k_hat, self.crb))

    def __repr__(self):
        if np.ndim(self.k_hat) == 0:
            head = f"k̂ = {float(self.k_hat):.6g} ± {float(self.crb):.2g} (CRB)"
        else:
            k = np.asarray(self.k_hat)
            c = np.asarray(self.crb)
            head = ("k̂ = [" + ", ".join(f"{v:.5g}" for v in k) + "] ± ["
                    + ", ".join(f"{v:.1g}" for v in c) + "] (CRB)")
        tail = f", '{self.verdict}'" if self.verdict else ""
        return f"Estimate({head}{tail})"


@dataclass
class Certificate:
    """A split-conformal error band with its finite-sample guarantee.

    P(relative error ≤ band) ≥ guarantee for a fresh random in-range query —
    by exchangeability alone. The guarantee is the HONEST reachable value
    (e.g. 8 calibration solves buy 8/9 ≈ 88.9%, not 90%).
    """
    band: float
    guarantee: float
    n_cal: int
    alpha: float
    scores: list = field(default_factory=list, repr=False)

    def __repr__(self):
        return (f"Certificate(err ≤ {self.band:.2e} with ≥{self.guarantee:.1%} "
                f"coverage; {self.n_cal} calibration solves)")
