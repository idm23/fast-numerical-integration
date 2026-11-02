"""Conserved quantities across the catalog.

- **Linear invariants** (``sum y`` for the chemical / decay chains) are
  preserved *exactly* by every Runge-Kutta method, up to round-off, at any
  tolerance. A loose tolerance must not change that.
- **Nonlinear invariants** (energy, angular momentum, Lotka-Volterra H,
  rigid-body Casimirs, Jacobi constant) drift, and the drift shrinks with the
  tolerance.
"""

import warnings

import numpy as np
import pytest

from fast_numerical_integration.integrators import solve_ivp
from fast_numerical_integration.problems import catalog

METHODS = ("RK23", "RK45", "DOP853")
LINEAR = ["B2", "B3", "C1", "C2", "robertson"]
NONLINEAR = ["B1", "B5", "D1", "D3", "D5", "C5", "pleiades", "arenstorf", "figure_eight_long"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", LINEAR)
def test_linear_invariants_exact_at_any_tolerance(name, method):
    p = catalog.get(name)
    (inv,) = p.invariants.values()
    for rtol in (1e-3, 1e-8):
        sol = solve_ivp(p.native(), p.t_span, p.y0, method=method, rtol=rtol, atol=rtol * 1e-3)
        vals = inv(sol.y)
        # Round-off relative to the largest magnitude reached. Robertson with
        # RK45 at rtol=1e-3 passes through |y| ~ 1e7 (an explicit-method
        # instability; see tests/stability/test_stiff.py) yet still conserves
        # the sum to ~1e-17 relative.
        magnitude = max(1.0, np.max(np.abs(sol.y)))
        assert np.max(np.abs(vals - vals[0])) < 1e-13 * sol.t.size * magnitude


@pytest.mark.parametrize("method", ["RK45", "DOP853"])
@pytest.mark.parametrize("name", NONLINEAR)
def test_nonlinear_invariant_drift_shrinks_with_tolerance(name, method):
    p = catalog.get(name)
    if name == "figure_eight_long":
        p = p.with_span(5 * catalog.figure_eight_system().period)
    drifts = {}
    for rtol in (1e-5, 1e-10):
        sol = solve_ivp(p.native(), p.t_span, p.y0, method=method, rtol=rtol, atol=rtol * 1e-3)
        drifts[rtol] = {}
        for inv_name, inv in p.invariants.items():
            vals = inv(sol.y)
            drifts[rtol][inv_name] = np.max(np.abs(vals - vals[0])) / max(abs(vals[0]), 1.0)
    for inv_name in p.invariants:
        loose, tight = drifts[1e-5][inv_name], drifts[1e-10][inv_name]
        assert tight < 1e4 * 1e-10, f"{inv_name}: {tight:.2e}"
        assert tight < max(loose / 100.0, 1e-13), f"{inv_name}: {loose:.2e} -> {tight:.2e}"
