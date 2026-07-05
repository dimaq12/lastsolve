"""
lastsolve — the last few solves you'll ever pay for.
================================================================================

An accelerator and identification layer over ANY black-box parametric
solver, built on the effective-rank dial of `resona` (the Spectra Without
Matrices series: measure the structure first, then pay accordingly).

    from lastsolve import accelerate, identify, audit

    @accelerate
    def solve(k): ...          # your expensive solver

    solve(0.021)               # ~10 real solves once, then microseconds,
    solve.stats                # with validated error and the Φ₁ dial

    identify(solve_fn, data, krange)   # k̂ ± Cramér–Rao bar, honest verdict
    audit(forward, x0, sigma, prior)   # how much answer your data contain

Design principles, in order: (1) never silently wrong — every answer is
validated, certified, or refused; (2) the dial before the fit — Φ₁ says
whether a cheap surrogate exists at all; (3) everything is counted — the
solver calls we spend are the price we quote.
"""
from .surrogate import Surrogate, TRANSFORMS
from .accelerate import accelerate, Accelerated
from .identify import identify, IdentifyResult
from .audit import audit, AuditReport
from .walls import detect_break, WallReport, classify_wall, WallClass, hard_points
from .multiparam import SurrogateND
from .adapters import CommandSolver, accelerate_command
from .timeprop import TimePropagator
from .spectral_id import identify_spectral
from .audit import normality_warning

__version__ = "1.0.0"
__all__ = [
    "Surrogate", "TRANSFORMS", "accelerate", "Accelerated",
    "identify", "IdentifyResult", "audit", "AuditReport", "normality_warning",
    "detect_break", "WallReport", "classify_wall", "WallClass", "hard_points",
    "SurrogateND", "CommandSolver", "accelerate_command",
    "TimePropagator", "identify_spectral", "__version__",
]
