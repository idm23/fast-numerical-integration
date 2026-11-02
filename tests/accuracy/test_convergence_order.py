"""Classical order of convergence with fixed step sizes.

With ``first_step = max_step = h`` and a huge tolerance the controller never
rejects and never shrinks, so the integrators run with a constant step ``h``.
The global error then scales as ``h**p`` with p = 3 (RK23), 5 (RK45) and
8 (DOP853). This is the standard order-verification test; a single wrong
tableau coefficient usually drops the observed order.
"""

import numpy as np
import pytest

from fast_numerical_integration import integrators
from fast_numerical_integration.integrators import solve_ivp
from fast_numerical_integration.problems import catalog

ORDERS = {"RK23": 3, "RK45": 5, "DOP853": 8}
PROBLEMS = ["B5", "A3", "D2", "E1"]


def _fixed_step_error(p, method, h, t_end):
    sol = solve_ivp(p.native(), (0.0, t_end), p.y0, method=method, rtol=1e3, atol=1e3,
                    first_step=h, max_step=h)
    assert np.allclose(np.diff(sol.t)[:-1], h)
    return np.max(np.abs(sol.y[:, -1] - p.exact(np.array([t_end]))[:, 0]))


@pytest.mark.parametrize("name", PROBLEMS)
@pytest.mark.parametrize("method", list(ORDERS))
def test_observed_order(method, name):
    """Halve h from 1.6 and fit the order over the asymptotic window.

    That window is the pairs of refinements whose errors sit between 1e-4
    (pre-asymptotic above) and 1e-11 (round-off below).
    """
    p = catalog.get(name)
    hs, errs = [], []
    h = 1.6
    while h > 1e-3:
        e = _fixed_step_error(p, method, h, t_end=8.0)
        hs.append(h)
        errs.append(e)
        if e < 1e-11:
            break
        h /= 2
    hs, errs = np.array(hs), np.array(errs)
    orders = np.diff(np.log(errs)) / np.diff(np.log(hs))
    window = (errs[:-1] < 1e-4) & (errs[1:] > 1e-11)
    usable = orders[window]
    assert usable.size >= 2, f"h={hs}, errors={errs}"
    observed = np.median(usable[-2:])
    assert abs(observed - ORDERS[method]) < 0.6, f"observed orders {np.round(usable, 2)}"


@pytest.mark.parametrize("method", list(ORDERS))
def test_dense_output_local_order(method):
    """One step of size h from the exact state, interpolant checked at the midpoint.

    The local interpolation error is O(h^(q+1)) for an interpolant of order q:
    q = 3 (RK23 cubic Hermite), 4 (RK45 quartic), 7 (DOP853).
    """
    p = catalog.get("B5")
    q = {"RK23": 3, "RK45": 4, "DOP853": 7}[method]
    cls = getattr(integrators, method)
    t0 = 0.3
    y0 = p.exact(np.array([t0]))[:, 0]
    hs = np.array([0.4, 0.2, 0.1, 0.05])
    errs = []
    for h in hs:
        solver = cls(p.native(), t0, y0, t0 + 10.0, rtol=1e3, atol=1e3, first_step=h, max_step=h)
        solver.step()
        tm = t0 + 0.37 * h
        errs.append(np.max(np.abs(solver.dense_output()(tm) - p.exact(np.array([tm]))[:, 0])))
    errs = np.array(errs)
    orders = np.diff(np.log(errs)) / np.diff(np.log(hs))
    usable = orders[errs[1:] > 1e-14]
    assert np.median(usable) > q + 1 - 0.5, f"observed local orders {np.round(orders, 2)}"
