"""
timeprop.py — the same bargain, applied to TIME: learn the propagator from
one trajectory, then step for free.

A trajectory you already computed is data about the dynamics itself. DMD /
Koopman theory says: one thin SVD turns the snapshot matrix into the
least-squares propagator. `resona.lift.koopman` provides the reduced
operator's action; its eigenvalues (read matrix-free by `resona.cloud`)
are the dynamic modes — |λ| > 1 means the learned dynamics are unstable
and long prediction is dishonest, and TimePropagator SAYS so.

Scope honesty: exact for linear dynamics, good for smooth dissipative ones
over modest horizons, and structurally unable to track chaos beyond the
Lyapunov time — the validation tail measures where you stand.
"""
from __future__ import annotations

import numpy as np
import resona
import resona.lift


class TimePropagator:
    """Learn u_{n+1} ≈ P u_n from a trajectory; predict with a receipt.

    snapshots : (n_times, n_features) — a single trajectory, uniform Δt.
    holdout   : fraction of the tail reserved for honest validation.
    """

    def __init__(self, snapshots, rank=None, holdout=0.2):
        S = np.asarray(snapshots, dtype=float)
        n_hold = max(2, int(len(S)*holdout))
        train, tail = S[:-n_hold], S[-n_hold:]
        X0, X1 = train[:-1].T, train[1:].T              # features × time
        U, sv, Vt = np.linalg.svd(X0, full_matrices=False)
        r = rank or int(np.sum(sv > 1e-12*sv[0]))
        U, sv, Vt = U[:, :r], sv[:r], Vt[:r]
        self.U = U
        self.Ktil = U.T @ X1 @ Vt.T @ np.diag(1.0/sv)   # r×r reduced propagator
        self.rank = r
        # resona reads: reduced action + its complex eigenvalue cloud
        mv, rmv, r2 = resona.lift.koopman(train.T, rank=r)
        eigs = np.linalg.eigvals(self.Ktil)             # r is small — exact
        self.mode_moduli = np.sort(np.abs(eigs))[::-1]
        self.stable = bool(self.mode_moduli[0] <= 1.0 + 1e-8)
        # honest validation on the held-out tail
        u = train[-1].copy()
        errs = []
        for y in tail:
            u = self.step(u)
            errs.append(np.linalg.norm(u-y)/max(np.linalg.norm(y), 1e-16))
        self.val_err = float(np.max(errs))
        self.horizon_validated = n_hold

    def step(self, u, n=1):
        """Advance the learned dynamics n steps: u ← U K̃ⁿ Uᵀ u."""
        z = self.U.T @ np.asarray(u, dtype=float)
        for _ in range(n):
            z = self.Ktil @ z
        return self.U @ z

    def __repr__(self):
        tag = "stable" if self.stable else "UNSTABLE — long prediction is dishonest"
        return (f"TimePropagator(rank={self.rank}, |λ|max={self.mode_moduli[0]:.4f} "
                f"[{tag}], val_err={self.val_err:.1e} over {self.horizon_validated} "
                f"held-out steps)")
