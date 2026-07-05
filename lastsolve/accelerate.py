"""
accelerate.py — the one-line speedup: a decorator that learns your solver's
parametric manifold and then answers from it.

    from lastsolve import accelerate

    @accelerate(span=0.35)
    def solve(k):
        ...            # your expensive solver — any code, any language behind it

The FIRST call at some k0 triggers an eager build: lastsolve spends the node
budget (default 7–31 Chebyshev solves + 3 validation solves, all through your
own function) on the range k0·(1±span), then serves every further in-range
call from the surrogate in microseconds. Out-of-range calls fall through to
the real solver and are counted — the decorator never silently extrapolates.

`solve.stats` tells you exactly what happened: real calls, served calls,
fallbacks, validated error, and the Φ₁ dial. Honesty is the interface.
"""
from __future__ import annotations

import functools
import numpy as np

from .surrogate import Surrogate


class Accelerated:
    """Callable wrapper produced by @accelerate."""

    def __init__(self, fn, span=0.35, warmup=1, ladder=(7, 11, 15, 23, 31),
                 target=5e-11, transform=None, seed=0):
        functools.update_wrapper(self, fn)
        self._fn = fn
        self._span = span
        self._warmup = max(1, int(warmup))
        self._seen = []                       # (k, y) during warmup
        self._ladder = ladder
        self._target = target
        self._transform = transform
        self._seed = seed
        self.surrogate = None
        self._real = 0
        self._served = 0
        self._fallback = 0

    def __repr__(self):
        s = self.stats
        if self.surrogate is None:
            return (f"Accelerated(warming up: {len(self._seen)}/{self._warmup} "
                    f"observed calls)")
        return (f"Accelerated({s['real_calls']} real → {s['served_from_surrogate']} "
                f"served, {s['fallbacks_out_of_range']} fallbacks; "
                f"range {s['range'][0]:.4g}..{s['range'][1]:.4g}, "
                f"val_err {s['validated_err']:.1e}, Φ₁ {s['phi1']:.2f})")

    def _real_call(self, k):
        self._real += 1
        return np.asarray(self._fn(float(k)), dtype=float)

    def __call__(self, k):
        k = float(k)
        if self.surrogate is None:
            if len(self._seen) + 1 < self._warmup:      # keep observing
                y = self._real_call(k)
                self._seen.append(k)
                return y
            self._seen.append(k)
            if len(self._seen) > 1:                     # range from what we SAW
                lo, hi = min(self._seen), max(self._seen)
                pad = 0.15*max(hi-lo, abs(0.5*(lo+hi))*0.05, 1e-12)
                ka, kb = lo-pad, hi+pad
            else:                                       # classic: span around k₀
                ka, kb = sorted((k*(1-self._span), k*(1+self._span)))
            s = Surrogate(lambda kk: self._fn(kk), (ka, kb),
                          transform=self._transform)
            s.fit(ladder=self._ladder, target=self._target, seed=self._seed)
            self._real += s.solves
            self.surrogate = s
            self._served += 1
            return s.query(k)
        if self.surrogate.ka <= k <= self.surrogate.kb:
            self._served += 1
            return self.surrogate.query(k)
        self._fallback += 1                      # honest: no extrapolation
        return self._real_call(k)

    @property
    def stats(self):
        s = self.surrogate
        return {
            "real_calls": self._real,
            "served_from_surrogate": self._served,
            "fallbacks_out_of_range": self._fallback,
            "range": None if s is None else (s.ka, s.kb),
            "nodes": None if s is None else s.n_nodes,
            "validated_err": None if s is None else s.val_err,
            "phi1": None if s is None else s.phi1,
        }


def accelerate(fn=None, **kw):
    """Decorator form: @accelerate or @accelerate(span=..., target=...)."""
    if fn is not None and callable(fn):
        return Accelerated(fn, **kw)
    return lambda f: Accelerated(f, **kw)
