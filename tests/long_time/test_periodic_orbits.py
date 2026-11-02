"""Periodic orbits integrated for several periods.

The figure-eight choreography is *linearly stable*, so integration errors stay
small for many periods. The Arenstorf orbit is *unstable*: perturbations,
including the published initial conditions' own truncation, grow
exponentially from period to period. Both behaviours are checked, with the
Jacobi constant as the invariant that must hold regardless.
"""

import numpy as np
import pytest

from fast_numerical_integration.integrators import solve_ivp
from fast_numerical_integration.problems import catalog
from fast_numerical_integration.three_body import figure_eight_system


def test_figure_eight_returns_each_period():
    fe = figure_eight_system()
    p = catalog.get("figure_eight_long")
    tq = fe.period * np.arange(0, 51)
    sol = solve_ivp(p.native(), p.t_span, p.y0, method="DOP853", rtol=1e-11, atol=1e-13, t_eval=tq)
    closure = np.max(np.abs(sol.y - p.y0[:, None]), axis=0)
    # The 8-digit initial conditions close the orbit to ~2e-6 per period; over
    # 50 periods the mismatch stays small (stable orbit, no exponential blow-up).
    assert closure[1] < 1e-5
    assert closure.max() < 1e-3
    E = p.invariants["energy"](sol.y)
    assert np.max(np.abs(E - E[0])) < 1e-8


def test_arenstorf_is_unstable_but_jacobi_constant_holds():
    p = catalog.get("arenstorf_long")
    T = catalog.ARENSTORF_PERIOD
    tq = T * np.arange(0, 6)
    sol = solve_ivp(p.native(), p.t_span, p.y0, method="DOP853", rtol=1e-12, atol=1e-14, t_eval=tq)
    closure = np.max(np.abs(sol.y - p.y0[:, None]), axis=0)[1:]
    assert closure[0] < 1e-8
    # Exponential growth: each period multiplies the closure error by >> 1.
    growth = closure[1:] / closure[:-1]
    assert np.all(growth > 5), f"per-period growth {growth}"
    C = catalog.jacobi_constant(sol.y)
    assert np.max(np.abs(C - C[0])) < 1e-8


@pytest.mark.parametrize("method", ["RK45", "DOP853"])
def test_arenstorf_closure_improves_with_tolerance(method):
    p = catalog.get("arenstorf")
    closure = []
    for rtol in (1e-6, 1e-9, 1e-12):
        sol = solve_ivp(p.native(), p.t_span, p.y0, method=method, rtol=rtol, atol=rtol * 1e-3)
        closure.append(np.max(np.abs(sol.y[:, -1] - p.y0)))
    assert closure[0] > 10 * closure[1] > 100 * closure[2]
