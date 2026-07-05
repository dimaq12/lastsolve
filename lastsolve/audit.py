"""
audit.py — before you reconstruct anything, measure how much answer your
data actually contain.

For FIELD-valued unknowns (a coefficient map ν(x), N unknowns) observed
through a black-box forward map, `audit` probes the Jacobian J = ∂obs/∂ν
matrix-free (2 solver calls per random direction), reads its effective rank
Φ₁ with resona, and counts the directions that rise above the noise. The
verdict is the one classical inversion pipelines and neural reconstructions
never give you: how many independent numbers the dataset holds, and how much
of the unknown is BLIND — unrecoverable by any method whatsoever.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import resona


@dataclass
class AuditReport:
    phi1: float
    n_probes: int
    solves: int
    sketch_singulars: np.ndarray
    visible: int
    verdict: str

    def __repr__(self):
        return (f"AuditReport(phi1={self.phi1:.2f}, visible≈{self.visible} of "
                f"{self.n_probes} probed directions, solves={self.solves}, "
                f"'{self.verdict}')")


def audit(forward, x0, sigma, prior_amp, n_probes=16, eps=None, seed=0):
    """Identifiability audit of a field-valued inverse problem.

    forward   : f(x: np.ndarray field) -> np.ndarray observation
    x0        : the prior / linearization point (field)
    sigma     : observation noise level (same units as the observation)
    prior_amp : expected deviation scale of the unknown field
    n_probes  : random directions to probe (2 solves each)

    Returns an AuditReport. `visible` counts sketch directions whose
    sensitivity beats sigma/prior_amp — a LOWER bound on the true visible
    count (a sketch can miss directions, never invent them). For exact
    counts build the full Jacobian: N columns × 2 solves.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    eps = eps or 1e-4*max(np.linalg.norm(x0)/np.sqrt(n), 1e-12)
    rng = np.random.default_rng(seed)
    solves = 0
    cols = []
    for _ in range(n_probes):
        d = rng.standard_normal(n)
        d /= np.linalg.norm(d)
        cols.append((np.asarray(forward(x0+eps*d), dtype=float)
                     - np.asarray(forward(x0-eps*d), dtype=float))/(2*eps))
        solves += 2
    G = np.array(cols).T                     # m × n_probes sketch of J
    m = G.shape[0]
    mv = lambda v: G @ (G.T @ v)             # GGᵀ matvec — resona's food
    phi1 = float(resona.of(mv, m, k=min(2*n_probes, m),
                           probes=128).effective_rank())
    s = np.linalg.svd(G, compute_uv=False)
    visible = int(np.sum(s > sigma/max(prior_amp, 1e-300)))
    if visible == 0:
        verdict = "the data contain NOTHING about this field at this noise level"
    else:
        verdict = (f"the data contain ≈{visible} independent numbers about "
                   f"{n} unknowns; the rest is blind for any method")
    return AuditReport(phi1=phi1, n_probes=n_probes, solves=solves,
                       sketch_singulars=s, visible=visible, verdict=verdict)
