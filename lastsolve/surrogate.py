"""
surrogate.py — the core of lastsolve: a certified Chebyshev surrogate of a
black-box parametric map k ↦ f(k).

You bring any expensive function f(k) -> np.ndarray (a PDE solver, a legacy
code behind a subprocess, a lab simulator). lastsolve spends a handful of
calls on Chebyshev nodes and returns:

  • query(k)   — the answer at ANY k in range, microseconds, near machine
                 precision when the parametric manifold is smooth;
  • phi1       — the effective rank of the response manifold, measured by
                 resona (the same dial as the rest of the Spectra Without
                 Matrices series): ~1 ⇒ healthy; ≫1 ⇒ a wall (do not trust
                 a linear-in-parameter surrogate); ~0 ⇒ parameter is dead;
  • deriv(k)   — the sensitivity ∂f/∂k (Fisher information for free);
  • invert(y)  — maximum-likelihood parameter identification with a
                 Cramér–Rao error bar, at zero extra solver calls
                 (optional polish on the true solver);
  • certify()  — split-conformal error band with a distribution-free,
                 finite-sample coverage guarantee.

Everything the surrogate spends is counted in .solves; every claim it makes
is measured, not asserted.
"""
from __future__ import annotations

import numpy as np
import resona
import resona.defect
from scipy.optimize import minimize_scalar

from .results import Certificate, Estimate, NotFittedError, OutOfRangeError

#: named parameter transforms for `transform='auto'` — the surrogate can
#: discover that a family is smooth in log k or 1/√k rather than k itself.
TRANSFORMS = {
    "identity": (lambda k: k, lambda p: p),
    "log": (np.log, np.exp),
    "sqrt": (np.sqrt, lambda p: p**2),
    "inverse": (lambda k: 1.0/k, lambda p: 1.0/p),
    "inv-sqrt": (lambda k: 1.0/np.sqrt(k), lambda p: 1.0/p**2),
}


class Surrogate:
    """Adaptive barycentric-Chebyshev surrogate of a scalar-parameter map.

    Parameters
    ----------
    forward : callable
        f(k: float) -> np.ndarray. The expensive black box. Called only
        during fit / certify / polish; every call is counted.
    krange : (float, float)
        The parameter interval the surrogate must cover.
    transform : None | (tf, tf_inv) | 'auto'
        Coordinate for the interpolation. 'auto' tries TRANSFORMS at 7
        nodes each and keeps the best validated one.
    """

    def __init__(self, forward, krange, transform=None):
        self._f_raw = forward
        self.solves = 0
        self.ka, self.kb = float(krange[0]), float(krange[1])
        self.transform_name = None
        if transform == 'auto':
            self.tf, self.tf_inv = self._auto_transform()
        elif transform is None:
            self.tf, self.tf_inv = TRANSFORMS["identity"]
            self.transform_name = "identity"
        else:
            self.tf, self.tf_inv = transform
            self.transform_name = "custom"
        pa, pb = self.tf(self.ka), self.tf(self.kb)
        self.a, self.b = min(pa, pb), max(pa, pb)
        self.fitted = False
        self.certificate = None

    def __repr__(self):
        if not self.fitted:
            return (f"Surrogate(unfitted, range [{self.ka:.4g}, {self.kb:.4g}], "
                    f"coordinate '{self.transform_name}')")
        cert = f", {self.certificate!r}" if self.certificate else ""
        return (f"Surrogate([{self.ka:.4g}, {self.kb:.4g}] in "
                f"'{self.transform_name}' coords: {self.n_nodes} nodes, "
                f"{self.solves} solves, val_err {self.val_err:.1e}, "
                f"Φ₁ {self.phi1:.2f}{cert})")

    # ── plumbing ─────────────────────────────────────────────────────────────
    def _f(self, k):
        self.solves += 1
        return np.asarray(self._f_raw(float(k)), dtype=float)

    def _auto_transform(self):
        """Pick the coordinate with the lowest 7-node validation error."""
        if self.ka <= 0:                       # log/inverse need k > 0
            candidates = {"identity": TRANSFORMS["identity"]}
        else:
            candidates = TRANSFORMS
        best, best_err = None, np.inf
        for name, (tf, tfi) in candidates.items():
            s = Surrogate(self._f_raw, (self.ka, self.kb), transform=(tf, tfi))
            s.fit(ladder=(7,), seed=7)
            self.solves += s.solves
            if s.val_err < best_err:
                best, best_err, best_name = (tf, tfi), s.val_err, name
        self.transform_name = best_name
        return best

    def _build(self, n, seed_offset=0):
        j = np.arange(n)
        x = np.cos((2*j+1)*np.pi/(2*n))
        self.xs = 0.5*(self.a+self.b) + 0.5*(self.b-self.a)*x
        self.ws = np.array([1.0/np.prod(self.xs[i]-np.delete(self.xs, i))
                            for i in range(n)])
        self.U = np.array([self._f(self.tf_inv(p)) for p in self.xs])
        self.n_nodes = n

    # ── fitting ──────────────────────────────────────────────────────────────
    def fit(self, ladder=(7, 11, 15, 23, 31), target=5e-11, seed=0):
        """Climb the node ladder until 3 held-out solves validate `target`."""
        rng = np.random.default_rng(seed)
        k_val = [self.ka + (self.kb-self.ka)*(0.1+0.8*r) for r in rng.random(3)]
        u_val = None
        for n in ladder:
            self._build(n)
            self.fitted = True            # nodes built ⇒ evaluable (validation below)
            if u_val is None:
                u_val = [self._f(k) for k in k_val]
            errs = [np.linalg.norm(self.query(k)-u)/max(np.linalg.norm(u), 1e-16)
                    for k, u in zip(k_val, u_val)]
            self.val_err = float(np.max(errs))
            if self.val_err < target:
                break
        self.fitted = True
        self._phi1 = None
        return self

    @property
    def phi1(self):
        """Effective rank Φ₁ of the centered snapshot covariance — computed
        matrix-free by resona, the series' measuring instrument."""
        if self._phi1 is None:
            Uc = self.U - self.U.mean(axis=0)
            m = Uc.shape[1]
            mv = lambda v: Uc.T @ (Uc @ v)
            self._phi1 = float(
                resona.of(mv, m, k=min(2*self.n_nodes, m),
                          probes=128).effective_rank())
        return self._phi1

    # ── queries ──────────────────────────────────────────────────────────────
    def _coef(self, p):
        d = p - self.xs
        i = int(np.argmin(np.abs(d)))
        if abs(d[i]) < 1e-13*max(abs(p), 1.0):
            return None, i
        return self.ws/d, None

    def query(self, k, strict=True):
        """The answer at k — microseconds. strict=True (default) refuses
        out-of-range queries with OutOfRangeError: lastsolve never
        extrapolates silently. strict=False extrapolates, on your head."""
        if not self.fitted:
            raise NotFittedError()
        tol = 1e-12*max(abs(self.ka), abs(self.kb), 1.0)
        if strict and not (self.ka - tol <= k <= self.kb + tol):
            raise OutOfRangeError(k, self.ka, self.kb)
        c, i = self._coef(self.tf(k))
        if c is None:
            return self.U[i]
        return (c @ self.U)/c.sum()

    __call__ = query

    def deriv(self, k):
        """∂f/∂k — analytic derivative of the interpolant (chain rule)."""
        if not self.fitted:
            raise NotFittedError()
        p = self.tf(k)
        c, i = self._coef(p)
        if c is None:
            p = p + 1e-9*max(abs(p), 1.0)
            c, _ = self._coef(p)
        d = p - self.xs
        cp = -c/d
        S0, S1 = c.sum(), c @ self.U
        dudp = (cp @ self.U - (S1/S0)*cp.sum())/S0
        ek = 1e-7*max(abs(k), 1.0)
        return dudp*(self.tf(k+ek)-self.tf(k-ek))/(2*ek)

    def sensitivity(self, k, refine=True, eps=None):
        """∂f/∂k measured on the TRUE solver by central differences.

        refine=True adds one Richardson step (resona.defect.richardson,
        p=2): two extra solves buy two extra orders of accuracy. Use this
        for Fisher information at a k where interpolation is not trusted —
        e.g. near a detected wall; inside the fitted range, `deriv(k)` is
        both free and more accurate.
        """
        e = eps or 1e-3*max(abs(k), 1e-3)
        W1 = (self._f(k+e) - self._f(k-e))/(2*e)
        if not refine:
            return W1
        W2 = (self._f(k+e/2) - self._f(k-e/2))/e
        return resona.defect.richardson(W1, W2, p=2)

    # ── inverse problem ──────────────────────────────────────────────────────
    def invert(self, y_obs, polish=False):
        """Maximum-likelihood k̂ from an observed field.

        polish=False: pure surrogate scan — ZERO calls to the real solver.
        polish=True : secant + projected Gauss–Newton on the true forward,
                      down to the float64 floor (clean data).
        Returns (k_hat, crb): crb = σ̂/‖∂f/∂k‖ with σ̂ estimated from the
        residual component orthogonal to the sensitivity — the Cramér–Rao
        bar, the floor no unbiased estimator can beat.
        """
        y_obs = np.asarray(y_obs, dtype=float)
        obj = lambda k: float(np.sum((self.query(k)-y_obs)**2))
        res = minimize_scalar(obj, bounds=(self.ka, self.kb), method='bounded',
                              options={'xatol': 1e-15})
        k_hat = float(res.x)
        W = self.deriv(k_hat)
        nW = np.linalg.norm(W)
        if polish and nW > 1e-12:
            w = W/nW
            phi = lambda kk: float(np.dot(w, self._f(kk)-y_obs))
            for pert in (1e-8, 1e-9):
                k1, f1 = k_hat, phi(k_hat)
                k2 = k_hat*(1+pert)+1e-14
                f2 = phi(k2)
                for _ in range(12):
                    if f2 == f1 or not np.isfinite(f2):
                        break
                    k3 = k2 - f2*(k2-k1)/(f2-f1)
                    if not np.isfinite(k3) or abs(k3-k2) < 1e-17*max(1.0, abs(k2)):
                        k2 = k3 if np.isfinite(k3) else k2
                        break
                    f3 = phi(k3)
                    if abs(f3) >= abs(f2):
                        break
                    k1, f1, k2, f2 = k2, f2, k3, f3
                k_hat = k2
            k_c, u_c = k_hat, self._f(k_hat)          # projected Gauss–Newton
            for _ in range(6):
                e = 1e-3*max(abs(k_c), 1e-3)
                Wt = (self._f(k_c+e)-self._f(k_c-e))/(2*e)
                WW = float(np.dot(Wt, Wt))
                if WW < 1e-28:
                    break
                g = float(np.dot(Wt, y_obs-u_c))/WW
                if not np.isfinite(g) or abs(g) < 1e-16*max(1.0, abs(k_c)):
                    break
                p_old = abs(float(np.dot(Wt, y_obs-u_c)))
                step, moved = g, False
                for _ in range(5):
                    k_try, u_try = k_c+step, None
                    u_try = self._f(k_try)
                    if abs(float(np.dot(Wt, y_obs-u_try))) < p_old:
                        k_c, u_c, moved = k_try, u_try, True
                        break
                    step *= 0.5
                if not moved:
                    break
            k_hat = k_c
            W = self.deriv(k_hat)
            nW = np.linalg.norm(W)
        r = y_obs - self.query(k_hat)
        if nW > 1e-12:
            w = W/nW
            r_perp = r - w*float(np.dot(w, r))
            sigma_hat = float(np.linalg.norm(r_perp)/np.sqrt(max(len(r)-1, 1)))
            crb = sigma_hat/nW
        else:
            crb = float('inf')
        rng_w = self.kb - self.ka
        if not np.isfinite(crb) or crb > 0.5*rng_w:
            verdict = "the data do not contain this parameter"
        elif crb > 0.05*rng_w:
            verdict = "weakly identifiable — an interval, not a number"
        else:
            verdict = "identifiable; the bar is the Cramér–Rao floor"
        return Estimate(k_hat=float(k_hat), crb=float(crb), verdict=verdict,
                        solves=self.solves)

    # ── certification ────────────────────────────────────────────────────────
    def certify(self, n_cal=8, alpha=0.1, seed=1):
        """Split-conformal error band with a finite-sample guarantee.

        Spends n_cal fresh solves at random k; the band is the appropriate
        order statistic of their relative errors. For a FRESH random query
        in range, P(error <= band) >= guarantee — by exchangeability alone,
        no smoothness assumptions. Note the honest finite-sample value: at
        n_cal=8, alpha=0.1 the reachable guarantee is 8/9 ≈ 88.9%, not 90%.
        """
        rng = np.random.default_rng(seed)
        scores = []
        for _ in range(n_cal):
            k = self.ka + (self.kb-self.ka)*rng.random()
            y = self._f(k)
            scores.append(np.linalg.norm(self.query(k)-y)
                          / max(np.linalg.norm(y), 1e-16))
        scores = np.sort(scores)
        rank = int(np.ceil((n_cal+1)*(1-alpha)))
        if rank > n_cal:
            band, guarantee = float(scores[-1]), n_cal/(n_cal+1)
        else:
            band, guarantee = float(scores[rank-1]), rank/(n_cal+1)
        self.certificate = Certificate(band=band, guarantee=guarantee,
                                       n_cal=n_cal, alpha=alpha,
                                       scores=list(scores))
        return self.certificate
