"""
adapters.py — accelerate solvers you cannot even import.

The legacy code is a binary? A Fortran relic behind a shell script? A
container? If you can run it from a command line, lastsolve can learn it:

    from lastsolve.adapters import CommandSolver, accelerate_command

    solver = CommandSolver("./mysolver --visc {k}", parser=parse_stdout)
    s = accelerate_command(solver, krange=(0.014, 0.026))
    s.query(0.0173)          # microseconds; the binary ran ~10 times total

The prototype behind this: a 200-point Burgers sweep served from 10
subprocess invocations, 20.8× faster end-to-end, answers matching real CLI
runs to 8.6·10⁻¹⁵.
"""
from __future__ import annotations

import shlex
import subprocess

import numpy as np

from .surrogate import Surrogate


class CommandSolver:
    """A black-box solver behind a shell command.

    cmd_template : e.g. "python3 solver.py --k {k}" — `{k}` is substituted.
    parser       : stdout (str) -> np.ndarray. Default: whitespace floats.
    Every invocation is counted in .calls — the price stays visible.
    """

    def __init__(self, cmd_template, parser=None, timeout=300):
        self.cmd_template = cmd_template
        self.parser = parser or (lambda out: np.fromstring(out, sep=' ')
                                 if False else np.array(out.split(), dtype=float))
        self.timeout = timeout
        self.calls = 0

    def __call__(self, k):
        self.calls += 1
        cmd = shlex.split(self.cmd_template.format(k=repr(float(k))))
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=self.timeout, check=True).stdout
        return self.parser(out)


def accelerate_command(solver, krange, transform=None, **fit_kw):
    """Fit a certified Surrogate over a CommandSolver (or any callable)."""
    s = Surrogate(solver, krange, transform=transform)
    s.fit(**fit_kw)
    return s
