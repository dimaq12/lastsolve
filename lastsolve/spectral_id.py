"""
spectral_id.py — identify a parameter through the operator's SPECTRUM.

When the field observable is useless — phase-fragile (solitons), rank-0
(the parameter barely moves the state), or you simply measure frequencies
rather than fields — the eigenvalues still carry the parameter. This module
inverts through them with resona's spectral-flow machinery:

  W_λ = ∂λ/∂k       — Hellmann–Feynman, via resona.wkernel.wkernel
  dk  = design step — resona.wkernel.design (W·dk ≈ λ_target − λ)
  λ polish          — resona.solve.rayleigh_polish, machine precision

This replaces the ad-hoc tridiagonal rescue used by the research stands
with the proper instrument.
"""
from __future__ import annotations

import numpy as np
import resona.solve
import resona.wkernel


def identify_spectral(family, k0, lam_targets, iters=10, fd_eps=1e-6,
                      tol=1e-13):
    """Find k such that the LOWEST eigenvalues of family(k) match a measured
    spectral fingerprint.

    family      : callable k -> H(k), a (small, symmetric) ndarray.
    k0          : starting parameter value.
    lam_targets : array of the m lowest observed eigenvalues (m >= 2
                  recommended — a single eigenvalue can be matched by more
                  than one k; the fingerprint disambiguates).
    Returns (k_hat, lam_hat): the parameter and the machine-polished
    eigenvalues actually achieved there.
    """
    lam_targets = np.atleast_1d(np.asarray(lam_targets, dtype=float))
    m = len(lam_targets)
    k = float(k0)
    lam_hat = None
    for _ in range(iters):
        H = np.asarray(family(k), dtype=float)
        evals, evecs = np.linalg.eigh(H)
        lam_hat = np.array([float(resona.solve.rayleigh_polish(H, ev))
                            for ev in evals[:m]])
        r = lam_targets - lam_hat
        if np.max(np.abs(r)) < tol*max(1.0, float(np.max(np.abs(lam_targets)))):
            break
        e = fd_eps*max(abs(k), 1.0)
        B = (np.asarray(family(k+e), dtype=float)
             - np.asarray(family(k-e), dtype=float))/(2*e)
        W = np.asarray(resona.wkernel.wkernel(evecs[:, :m], [B]),
                       dtype=float).reshape(m, 1)   # ∂λ_i/∂k, Hellmann–Feynman
        dk = resona.wkernel.design(W, r)            # LSQ over the fingerprint
        step = float(np.ravel(dk)[0])
        if not np.isfinite(step) or step == 0.0:
            break
        k = k + step
    return k, lam_hat
