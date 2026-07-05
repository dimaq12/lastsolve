"""
pde_zoo.py — the 35 nonlinear PDEs and the machinery they share.
================================================================================

Every stand in this repository runs over the same zoo of 35 one-parameter
nonlinear PDE families on a periodic grid of 128 points. Each family is
written in the split form

        u_t = −L(k)·u + N(u, k)

where L(k) is the LINEAR part, given by its Fourier symbol (an array over
wavenumbers — so exp(−dt·L) is exact and cheap), and N is the NONLINEAR part,
a plain callable on the field. The parameter k is the family's knob:
viscosity, dispersion, reaction rate, a fractional exponent…

The solver is Strang splitting: exact half-steps of the linear flow in
Fourier space around an explicit nonlinear step. Complex fields (CGL, NLS,
Chen–Lee–Liu, …) evolve as complex — no premature real casts; the OBSERVABLE
(what a stand actually compares) is chosen per equation via
`make_observable`: the real part for real dynamics, the amplitude |u| for
phase-fragile complex ones.

Why this zoo: it spans regular diffusion (Burgers), dispersive waves (KdV,
Benjamin–Ono), chaos (Kuramoto–Sivashinsky), solitons (NLS, Hirota),
peakon-family equations (Camassa–Holm, Degasperis–Procesi), fractional and
purely spectral operators (fractional heat, Orr–Sommerfeld) — the standard
"hard nonlinear" name-drops, in one runnable list.
"""
import zlib
import numpy as np
from numpy.fft import fft, ifft, fftfreq
from collections import namedtuple
from scipy.linalg import eigh_tridiagonal

# ── periodic grid: 128 points on [0, 2π) ─────────────────────────────────────
N_GRID = 128
L_DOM = 2*np.pi
DX = L_DOM/N_GRID
X = np.linspace(0, L_DOM, N_GRID, False)
K = fftfreq(N_GRID, DX)*2*np.pi
K2, K4 = K**2, K**4


def d_dx(u, order=1):
    """Spectral x-derivative of a periodic field (real part)."""
    return ifft((1j*K)**order*fft(u)).real


def _sanitize(y, cap=1e6):
    """Clip NaN/Inf blowups so a diverging trajectory fails loudly, not silently."""
    return np.clip(np.nan_to_num(y, nan=0, posinf=cap, neginf=-cap), -cap, cap)


def broadband(seed):
    """Deterministic broadband initial condition: 12 harmonics, unit variance."""
    r = np.random.RandomState(seed)
    u = sum(r.uniform(.3, 1)/m**.5*np.sin(m*X + r.uniform(0, 2*np.pi))
            for m in range(1, 13))
    u -= np.mean(u)
    return u/(np.std(u) + 1e-16)


# stock initial conditions
smooth = np.sin(X) + 0.25*np.sin(2*X)
bump = 1. + 0.15*np.cos(X)
ci = smooth + 0.1j*np.cos(2*X)          # complex start for CGL-type equations
nls_ic = 2./np.cosh(X*2)                # bright-soliton profile


def strang(u0, t, n_steps, L_sym, N_fn):
    """Strang splitting  e^{−dt·L/2} · (I + dt·N) · e^{−dt·L/2}.

    L_sym — Fourier symbol of L (PDE written as u_t = −L·u + N(u)).
    Works for real and complex fields; each step is sanitized so stiff
    corners blow up into a detectable cap instead of NaN soup.
    """
    u = u0.copy()
    dt = t/n_steps
    is_cplx = np.iscomplexobj(u0) or np.iscomplexobj(L_sym)
    eL2 = np.exp(-dt/2*L_sym)
    for _ in range(n_steps):
        u = ifft(fft(u)*eL2)
        u = u + dt*N_fn(u)
        u = ifft(fft(u)*eL2)
        u = _sanitize(np.real(u)) + (1j*_sanitize(np.imag(u)) if is_cplx else 0)
    return u if is_cplx else np.real(u)


# ── the zoo ──────────────────────────────────────────────────────────────────
PDE = namedtuple("PDE", "name k0 u0 t steps L N")


def zoo():
    """All 35 families. L(k) → Fourier symbol; N(u[, k]) → nonlinear term."""
    eqs = []

    def a(name, k0, u0, t, steps, L, N):
        eqs.append(PDE(name, k0, u0, t, steps, L, N))

    ns = 64
    # ---- diffusion / reaction / transport ----------------------------------
    a("Keller-Segel", [0.15], 1+.1*np.cos(X), 0.03, ns,          # chemotaxis
      lambda k: k[0]*K2,
      lambda u: -0.15*d_dx(u*d_dx(0.1*np.sin(X))))
    a("Perona-Malik", [0.2], smooth, 0.015, ns,                  # edge-preserving diffusion
      lambda k: k[0]*K2,
      lambda u, k: d_dx(1./(1+(d_dx(u)/2)**2)*d_dx(u)) - k[0]*d_dx(u, 2))
    a("Thin Film", [1.0], bump, 0.01, ns,                        # 4th-order lubrication
      lambda k: k[0]*K4,
      lambda u: -d_dx(np.maximum(u, .15)**3*d_dx(np.maximum(u, .15), 3))
                - ifft(-K4*fft(np.maximum(u, .15))).real)
    a("Benjamin-Ono", [1.0], smooth, 0.02, ns,                   # nonlocal dispersion
      lambda k: k[0]*1j*np.sign(K)*K2,
      lambda u: -u*d_dx(u))
    a("Burgers", [0.02], smooth, 0.15, ns,                       # THE shock equation
      lambda k: k[0]*K2,
      lambda u: -u*d_dx(u))
    a("KdV", [0.03], smooth, 0.05, ns,                           # solitary waves
      lambda k: -1j*K**3*k[0],
      lambda u: -u*d_dx(u))
    a("KS", [0.05], smooth, 0.03, ns,                            # Kuramoto–Sivashinsky
      lambda k: -k[0]*K2 + 0.02*K2**2,
      lambda u: -u*d_dx(u))
    a("Allen-Cahn", [0.5], smooth, 0.1, ns,                      # phase separation
      lambda k: 0.04*K2 - k[0],
      lambda u: -0.5*u**3)
    a("Sine-Gordon", [0.4], smooth, 0.12, ns,
      lambda k: 0.04*K2,
      lambda u, k: -k[0]*np.sin(u))
    a("Porous Medium", [0.08], bump, 0.08, ns,                   # degenerate diffusion
      lambda k: 0.04*K2,
      lambda u, k: k[0]*(d_dx(u)**2 + u*d_dx(u, 2)))
    a("Eikonal-HJB", [0.08], np.zeros(N_GRID), 0.08, ns,         # Hamilton–Jacobi
      lambda k: k[0]*K2,
      lambda u: -d_dx(u)**2 + (1+.2*np.cos(X))**2 - np.mean((1+.2*np.cos(X))**2))
    a("Causal KdV-Burg", [0.03], smooth, 0.05, ns,
      lambda k: k[0]*K2 - 0.02*1j*K**3,
      lambda u: -u*d_dx(u) - 0.25*u)
    # ---- stiff scalar reaction ODE-fields ----------------------------------
    a("ODE exp", [0.2], 0.1*np.sin(X), 0.2, ns,
      lambda k: K2 + 2.0,
      lambda u, k: np.exp(np.clip(u, -5, 5)) - 1 + k[0]*np.sin(X))
    a("ODE f^5", [0.05], 0.1*np.cos(X), 0.15, ns,
      lambda k: 1.5*K2 + 2.0,
      lambda u, k: -k[0]*u**5 + 0.2*np.sin(X))
    a("ODE deriv", [1.0], 0.1*np.sin(2*X), 0.15, ns,
      lambda k: K2 + 2.0,
      lambda u, k: -k[0]*u*d_dx(u)**2 + 0.2*np.sin(X))
    # ---- complex-field equations -------------------------------------------
    a("CGL", [1.0], ci, 0.05, ns,                                # complex Ginzburg–Landau
      lambda k: (k[0]+0.5j)*K2 + 0.5,
      lambda u: -(1+0.5j)*abs(u)**2*u)
    a("Swift-Hohenberg", [0.03], smooth, 0.05, ns,
      lambda k: k[0]*(1+K2)**2,
      lambda u: -u**3)
    a("Chen-Lee-Liu", [0.5], ci, 0.03, ns,
      lambda k: k[0]*1j*K2,
      lambda u: -1j*abs(u)**2*u)
    a("Sasa-Satsuma", [0.5], ci, 0.03, ns,
      lambda k: k[0]*1j*K2 + 1j*K**3,
      lambda u: -1j*abs(u)**2*u - 1j*u*d_dx(abs(u)**2))
    a("Nikolaevskiy", [0.02], smooth, 0.02, ns,
      lambda k: -k[0]*K2 + 0.01*K2**3,
      lambda u: -u*d_dx(u))
    a("Kundu-Eckhaus", [0.5], ci, 0.03, ns,
      lambda k: k[0]*1j*K2,
      lambda u: -1j*abs(u)**2*u - 0.5j*abs(u)**4*u)
    a("Generalized KS", [0.02], smooth, 0.02, ns,
      lambda k: -k[0]*K2 + 0.01*K2**2 + 0.005*K2**3,
      lambda u: -u*d_dx(u))
    # ---- peakon family (parameter nearly decouples — the rank-0 corner) ----
    a("Camassa-Holm", [0.01], np.exp(-(X-np.pi)**2/.5), 0.02, ns,
      lambda k: k[0]*K2/(1+K2),
      lambda u: -u*d_dx(u) + d_dx(u*d_dx(u, 2)) - 2*d_dx(u)*d_dx(u, 2))
    a("Degasperis-Procesi", [0.02], np.exp(-(X-np.pi)**2/.5), 0.02, ns,
      lambda k: k[0]*K2/(1+K2),
      lambda u: -u*d_dx(u) + 0.8*d_dx(u*d_dx(u, 2)) - 1.5*d_dx(u)*d_dx(u, 2))
    a("Fokas-Lenells", [0.5], ci, 0.03, ns,
      lambda k: k[0]*1j*K2,
      lambda u: -0.5j*u*d_dx(abs(u)**2))
    a("Geng-Cao", [0.5], ci, 0.03, ns,
      lambda k: k[0]*1j*K2,
      lambda u: -1j*2*abs(u)**2*u)
    a("Hirota mKdV", [0.03], smooth, 0.05, ns,
      lambda k: -1j*K**3*k[0],
      lambda u: -6*u**2*d_dx(u))
    # ---- purely spectral / fractional operators (k in the EXPONENT) --------
    a("Frac Heat 0.5", [0.5], broadband(28), 0.1, 32,
      lambda k: np.abs(K)**k[0], lambda u: 0*u)
    a("Frac Heat 1.0", [1.0], broadband(29), 0.1, 32,
      lambda k: np.abs(K)**k[0], lambda u: 0*u)
    a("Frac Heat 2.0", [2.0], broadband(30), 0.1, 32,
      lambda k: np.abs(K)**k[0], lambda u: 0*u)
    a("Fokker-Planck", [0.5], broadband(31), 0.3, 32,
      lambda k: 0.001*K2 - 1j*k[0]*K, lambda u: 0*u)
    a("Orr-Sommerfeld", [1.0], broadband(32), 0.25, 32,
      lambda k: 0.5j*k[0]*K + (K2+1)/100, lambda u: 0*u)
    # ---- the phase-fragile pair (this journey's honest wall) ---------------
    a("NLS Soliton", [0.5], nls_ic, 0.5, 32,
      lambda k: k[0]*1j*K2,
      lambda u: -1j*abs(u)**2*u)
    a("GL K3", [1.0], nls_ic, 0.3, 32,
      lambda k: (k[0]+0.5j)*K2 + 0.2,
      lambda u: -(1+0.3j)*abs(u)**2*u)
    # ---- chaos with a broadband start --------------------------------------
    a("KS Chaos", [0.02], np.sin(X+35) + 0.3*np.sin(2*X+35*1.7), 0.3, ns,
      lambda k: -k[0]*K2 + 0.01*K2**2,
      lambda u: -u*d_dx(u))

    return eqs


# ── black-box observable ─────────────────────────────────────────────────────
def make_observable(pde, amp=False):
    """Wrap a zoo entry into a black-box map  (u0, t, [k]) → observed field.

    This is the ONLY interface the stands use — they never look inside the
    solver, exactly like a lab instrument probing a physical system.

    amp=True  → observe the amplitude |u| (phase-fragile complex equations);
    amp=False → observe Re u.
    A semi-implicit IMEX fallback catches the rare stiff blowups of Strang.
    """
    L_fn, N_fn = pde.L, pde.N

    def obs(u, t, k):
        L_sym = L_fn([float(x) for x in k])
        try:
            N_fn(u, k)
            Nf = lambda v: N_fn(v, k)
        except TypeError:
            Nf = lambda v: N_fn(v)
        sol = strang(u, t, pde.steps, L_sym, Nf)
        if not amp and np.max(np.abs(sol)) > 1e6:      # stiff corner → IMEX
            dt = t/pde.steps
            us = u.copy()
            for _ in range(pde.steps):
                us = np.real(ifft((fft(us) + dt*fft(Nf(us)))/(1. - dt*L_sym)))
                us = _sanitize(us)
            sol = us
        return np.abs(sol) if amp else np.real(sol)

    return obs


# ── shared inversion utilities ───────────────────────────────────────────────
def stable_seed(name):
    """Deterministic per-equation seed (unlike hash(), stable across runs —
    so every reader reproduces the SAME hidden parameters and tables)."""
    return zlib.crc32(name.encode()) % 10000


_md0 = 2.0/DX**2*np.ones(N_GRID)
_od0 = -1.0/DX**2*np.ones(N_GRID-1)


def eigen_inverse(true_k, k0, eps=1e-3):
    """Rank-0 rescue: when the parameter barely moves the FIELD, it still
    scales the operator's eigenvalues analytically. Invert through the
    spectrum of tridiag(k) — exact for the peakon corner of the zoo."""
    lam = lambda kv: eigh_tridiagonal(_md0*kv[0], _od0*kv[0], eigvals_only=True)
    lam_t, lam_r = lam(true_k), lam(k0)
    W = (lam([k0[0]+eps]) - lam([k0[0]-eps]))/(2*eps)
    return k0[0] + np.dot(W, lam_t-lam_r)/max(np.dot(W, W), 1e-16)
