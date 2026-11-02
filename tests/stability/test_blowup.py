"""Finite-time blow-up: ``y' = y**2`` with ``y(0) = 1`` is singular at ``t = 1``.

A correct adaptive integrator must notice that it can't continue: steps shrink
toward the singularity until they fall below the floating-point spacing of
``t``, then it stops with status -1. It must not step over the pole and
report success.

The numerical solution blows up at a slightly perturbed time ``1 + O(tol)``:
the discrete RK map has its own pole, offset from the exact one by the local
error. The stopping time converges to 1 as the tolerance tightens.

Exact comparisons (RHS evaluation counts, step sequences) are made against
SciPy run with our summation order (``testing.matched_solve_ivp``), which is
bit-identical to our engine on every platform. Stock SciPy's rounding varies
with the platform's SIMD kernels and C runtime, so its exact counts can move
in rounding-sensitive regimes (seen on Linux aarch64, macOS and Windows CI).
"""

import numpy as np
import pytest

from fast_numerical_integration.integrators import solve_ivp
from fast_numerical_integration.problems import catalog
from fast_numerical_integration.problems.native import Quadratic
from fast_numerical_integration.testing import matched_solve_ivp

METHODS = ("RK23", "RK45", "DOP853")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("rtol", [1e-3, 1e-6, 1e-10])
def test_fails_cleanly_at_the_singularity(method, rtol):
    sol = solve_ivp(Quadratic(), (0.0, 2.0), [1.0], method=method, rtol=rtol, atol=rtol * 1e-3)
    assert not sol.success
    assert sol.status == -1
    assert "step size" in sol.message
    t_stop = sol.t[-1]
    assert abs(t_stop - 1.0) < 10 * rtol
    assert sol.y[0, -1] > 100.0
    # Before stopping, the solution tracks the exact 1 / (1 - t).
    ok = sol.t < 0.99
    np.testing.assert_allclose(sol.y[0, ok], 1.0 / (1.0 - sol.t[ok]), rtol=1e4 * rtol)


@pytest.mark.parametrize("method", METHODS)
def test_stops_where_scipy_stops(method):
    p = catalog.get("blowup")
    kw = dict(method=method, rtol=1e-8, atol=1e-11)
    ref = matched_solve_ivp(p.rhs, p.t_span, p.y0, **kw)
    sol = solve_ivp(p.native(), p.t_span, p.y0, **kw)
    assert sol.status == ref.status == -1
    assert sol.message == ref.message
    assert sol.nfev == ref.nfev
    assert sol.t[-1] == pytest.approx(ref.t[-1], rel=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_terminal_event_catches_the_blowup_first(method):
    """A guard event (y = 1e6) turns the failure into a clean, accurate stop."""
    from fast_numerical_integration.integrators import ComponentCrossing

    guard = ComponentCrossing(0, 1e6, terminal=True)
    sol = solve_ivp(Quadratic(), (0.0, 2.0), [1.0], method=method, rtol=1e-10, atol=1e-12, events=guard)
    assert sol.status == 1
    assert sol.t_events[0][0] == pytest.approx(1.0 - 1e-6, rel=1e-8)
