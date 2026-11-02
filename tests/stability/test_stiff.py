"""Stiff problems with explicit methods: Van der Pol for increasing mu, Robertson.

Explicit methods still solve stiff problems, at a cost that grows with the
stiffness instead of the accuracy asked for. The tell-tale signature checked
here is that the RHS evaluation count scales with mu and stops depending on
the tolerance.
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp as scipy_solve_ivp

from fast_numerical_integration.integrators import solve_ivp
from fast_numerical_integration.problems import catalog
from fast_numerical_integration.testing import scipy_with_our_summation_order

METHODS = ("RK23", "RK45", "DOP853")


def _nfev(name, method, rtol):
    p = catalog.get(name)
    sol = solve_ivp(p.native(), p.t_span, p.y0, method=method, rtol=rtol, atol=rtol * 1e-3)
    assert sol.success
    return sol.nfev


@pytest.mark.parametrize("method", METHODS)
def test_vdp_cost_scales_with_stiffness(method):
    n100 = _nfev("vdp_mu100", method, 1e-6)
    n1000 = _nfev("vdp_mu1000", method, 1e-6)
    # Stability-limited step ~ 1/mu  =>  ~10x the work for 10x the stiffness.
    assert 6 < n1000 / n100 < 15


@pytest.mark.parametrize("method", METHODS)
def test_vdp_mu1000_cost_independent_of_tolerance(method):
    loose, tight = _nfev("vdp_mu1000", method, 1e-3), _nfev("vdp_mu1000", method, 1e-7)
    assert tight < 1.2 * loose


@pytest.mark.parametrize("method", METHODS)
def test_vdp_nonstiff_cost_does_depend_on_tolerance(method):
    loose, tight = _nfev("vdp_mu1", method, 1e-3), _nfev("vdp_mu1", method, 1e-9)
    assert tight > 3 * loose


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mu", [1, 10, 100, 1000])
def test_vdp_accuracy(method, mu):
    p = catalog.get(f"vdp_mu{mu}")
    rtol = 1e-7
    sol = solve_ivp(p.native(), p.t_span, p.y0, method=method, rtol=rtol, atol=rtol * 1e-3)
    assert catalog.end_error(p, sol.y[:, -1]) < 1e5 * rtol


@pytest.mark.parametrize("method", METHODS)
def test_robertson_matches_radau(method):
    p = catalog.get("robertson")
    rtol = 1e-7
    sol = solve_ivp(p.native(), p.t_span, p.y0, method=method, rtol=rtol, atol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        radau = scipy_solve_ivp(p.rhs, p.t_span, p.y0, method="Radau", rtol=1e-11, atol=1e-16)
    np.testing.assert_allclose(sol.y[:, -1], radau.y[:, -1], rtol=1e-5, atol=1e-12)
    assert np.all(sol.y > -1e-10)  # concentrations stay (numerically) non-negative


def test_rk45_loose_tolerance_instability_on_robertson_matches_scipy():
    """Documented explicit-RK pathology, reproduced exactly.

    At rtol=1e-3 RK45 accepts a run of steps in which y2 and y3 oscillate with
    growing amplitude (up to |y| ~ 1e7) while their sum stays exactly
    conserved: the embedded error estimate does not see the instability on
    this stiff problem (Hairer & Wanner, vol. II, on stiffness detection in
    Dormand-Prince). Eventually the step size collapses and the run fails.
    SciPy's RK45 does the same. RK23 and DOP853, and RK45 at tighter
    tolerances, do not.

    *When* the instability starts depends on rounding: the controller sits on
    the stability boundary, where it is driven by round-off. SciPy's own onset
    time moves (t ~ 0.34 vs ~ 0.30) just from swapping the NumPy RHS for the
    compiled one. With our summation order, SciPy reproduces our run bit for
    bit.
    """
    p = catalog.get("robertson")
    kw = dict(method="RK45", rtol=1e-3, atol=1e-6)
    ours = solve_ivp(p.native(), p.t_span, p.y0, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = scipy_solve_ivp(p.rhs, p.t_span, p.y0, **kw)
        with scipy_with_our_summation_order():
            matched = scipy_solve_ivp(p.native(), p.t_span, p.y0, **kw)

    for r in (ours, ref):
        assert r.status == -1 and "step size" in r.message
        assert np.abs(r.y).max() > 1e6
        y = np.abs(r.y).max(axis=0)
        onset = np.argmax(y > 1.5)
        assert 0.1 < r.t[onset] < 1.0
        # Mass is still conserved to round-off relative to the excursion.
        assert np.max(np.abs(r.y.sum(axis=0) - 1.0)) < 1e-13 * y.max() * r.t.size

    assert matched.nfev == ours.nfev
    np.testing.assert_array_equal(matched.t, ours.t)
    np.testing.assert_array_equal(matched.y, ours.y)

    for method in ("RK23", "DOP853"):
        clean = solve_ivp(p.native(), p.t_span, p.y0, method=method, rtol=1e-3, atol=1e-6)
        assert clean.success and np.abs(clean.y).max() < 1.5
    tight = solve_ivp(p.native(), p.t_span, p.y0, method="RK45", rtol=1e-6, atol=1e-9)
    assert tight.success and np.abs(tight.y).max() < 1.5
