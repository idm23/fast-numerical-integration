"""Linear stability: the step size on stiff problems is set by the stability region.

For ``y' = lambda y`` an explicit RK method multiplies the solution by the
stability polynomial ``R(h lambda)`` each step, and is stable only while
``|R(h lambda)| <= 1``. On the real axis that holds for ``-x* <= h lambda <= 0``.

The Prothero-Robinson problem ``y' = lambda (y - cos t) - sin t`` has the smooth
solution ``cos t``. The accuracy requirement alone would allow huge steps, so
the controller settles on the stability boundary: ``|h lambda|`` hovers around
``x*``. Here ``x*`` is computed directly from each Butcher tableau and checked
against the literature (2.51 for Bogacki-Shampine, 3.31 for Dormand-Prince).
"""

import numpy as np
import pytest
from numpy.polynomial import polynomial as P

from fast_numerical_integration import tableaus
from fast_numerical_integration.integrators import solve_ivp
from fast_numerical_integration.problems import catalog
from fast_numerical_integration.problems.native import ProtheroRobinson

METHODS = ("RK23", "RK45", "DOP853")


def _tableau(method):
    if method == "RK23":
        return tableaus.RK23_A, tableaus.RK23_B
    if method == "RK45":
        return np.pad(tableaus.RK45_A, ((0, 0), (0, 1))), tableaus.RK45_B
    ns = tableaus.DOP853_N_STAGES
    return tableaus.DOP853_A[:ns, :ns], tableaus.DOP853_B


def stability_polynomial(method):
    """Coefficients of R(z) = 1 + z b^T (I - zA)^{-1} 1 for explicit (nilpotent) A."""
    A, B = _tableau(method)
    s = len(B)
    coeffs = [1.0]
    Ak = np.eye(s)
    for _ in range(s):
        coeffs.append(B @ Ak @ np.ones(s))
        Ak = Ak @ A
    return np.array(coeffs)


def real_stability_boundary(method):
    coeffs = stability_polynomial(method)
    xs = np.linspace(1e-3, 12.0, 400001)
    return xs[np.argmax(np.abs(P.polyval(-xs, coeffs)) > 1.0)]


@pytest.mark.parametrize("method,order", [("RK23", 3), ("RK45", 5), ("DOP853", 8)])
def test_stability_polynomial_matches_exp_to_order(method, order):
    """R(z) agrees with e^z through z^order: the tableau's order conditions."""
    from math import factorial

    coeffs = stability_polynomial(method)
    for k in range(order + 1):
        assert coeffs[k] == pytest.approx(1.0 / factorial(k), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("method,literature", [("RK23", 2.51), ("RK45", 3.31)])
def test_real_stability_boundary_matches_literature(method, literature):
    assert real_stability_boundary(method) == pytest.approx(literature, abs=0.01)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("lam", [-1e2, -1e3, -1e4])
def test_step_size_rides_the_stability_boundary(method, lam):
    x_star = real_stability_boundary(method)
    sol = solve_ivp(ProtheroRobinson(lam), (0.0, 10.0), [1.0], method=method, rtol=1e-3, atol=1e-6)
    assert sol.success
    hl = np.abs(np.diff(sol.t)) * abs(lam)
    late = hl[len(hl) // 2:]
    assert np.median(late) == pytest.approx(x_star, rel=0.05)
    # Riding the boundary is stable: the smooth solution is still tracked.
    assert np.max(np.abs(sol.y[0] - np.cos(sol.t))) < 1e-2
    # The work therefore scales with |lambda|, not with the tolerance.
    assert sol.nfev > 0.5 * 10.0 * abs(lam) / x_star


@pytest.mark.parametrize("method", METHODS)
def test_work_is_independent_of_tolerance_when_stability_limited(method):
    p = catalog.get("linear_stiff")
    nfev = [solve_ivp(p.native(), p.t_span, p.y0, method=method, rtol=r, atol=r * 1e-3).nfev
            for r in (1e-2, 1e-4)]
    assert nfev[1] < 1.3 * nfev[0]


@pytest.mark.parametrize("method", METHODS)
def test_oscillatory_eigenvalues_stay_stable(method):
    """lambda = -1 +/- 100i: imaginary-axis stability; tracks the exact solution."""
    p = catalog.get("damped_oscillator")
    rtol = 1e-6
    sol = solve_ivp(p.native(), p.t_span, p.y0, method=method, rtol=rtol, atol=rtol * 1e-3)
    assert catalog.end_error(p, sol.y[:, -1]) < 1e4 * rtol
