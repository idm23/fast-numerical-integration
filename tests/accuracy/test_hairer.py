"""Hairer-Norsett-Wanner benchmark problems: Arenstorf orbit, Pleiades, Brusselator."""

import numpy as np
import pytest

from fast_numerical_integration.integrators import solve_ivp
from fast_numerical_integration.problems import catalog

METHODS = ("RK23", "RK45", "DOP853")


@pytest.mark.parametrize("rtol,bound", [(1e-6, 1e-2), (1e-9, 1e-5), (1e-12, 1e-8)])
def test_arenstorf_orbit_closes_after_one_period(rtol, bound):
    """The canonical DOP853 demo: return to y0 after T = 17.0652..."""
    p = catalog.get("arenstorf")
    sol = solve_ivp(p.native(), p.t_span, p.y0, method="DOP853", rtol=rtol, atol=rtol * 1e-3)
    assert np.max(np.abs(sol.y[:, -1] - p.y0)) < bound


def test_arenstorf_lower_order_methods_need_far_more_work():
    p = catalog.get("arenstorf")
    nfev = {m: solve_ivp(p.native(), p.t_span, p.y0, method=m, rtol=1e-8, atol=1e-11).nfev
            for m in METHODS}
    assert nfev["DOP853"] < nfev["RK45"] < nfev["RK23"]
    assert nfev["RK23"] > 10 * nfev["DOP853"]


@pytest.mark.parametrize("method", METHODS)
def test_arenstorf_jacobi_constant(method):
    p = catalog.get("arenstorf")
    rtol = 1e-8
    sol = solve_ivp(p.native(), p.t_span, p.y0, method=method, rtol=rtol, atol=rtol * 1e-3)
    C = catalog.jacobi_constant(sol.y)
    assert np.max(np.abs(C - C[0])) < 1e4 * rtol


@pytest.mark.parametrize("method", ["RK45", "DOP853"])
def test_pleiades(method):
    """Seven bodies with close encounters: steps vary over orders of magnitude."""
    p = catalog.get("pleiades")
    rtol = 1e-8
    sol = solve_ivp(p.native(), p.t_span, p.y0, method=method, rtol=rtol, atol=rtol * 1e-3)
    assert catalog.end_error(p, sol.y[:, -1]) < 1e4 * rtol
    h = np.diff(sol.t)
    assert h.max() / h.min() > 100
    for name, inv in p.invariants.items():
        vals = inv(sol.y)
        assert np.max(np.abs(vals - vals[0])) < 1e4 * rtol * max(1.0, abs(vals[0])), name


@pytest.mark.parametrize("method", METHODS)
def test_brusselator_limit_cycle(method):
    p = catalog.get("brusselator")
    rtol = 1e-7
    sol = solve_ivp(p.native(), p.t_span, p.y0, method=method, rtol=rtol, atol=rtol * 1e-3)
    assert catalog.end_error(p, sol.y[:, -1]) < 1e4 * rtol
    # Trajectory settles onto the limit cycle around the fixed point (A, B/A) = (1, 3).
    late = sol.y[:, sol.t > 10]
    assert late[0].min() < 1.0 < late[0].max()
    assert late[1].min() < 3.0 < late[1].max()
