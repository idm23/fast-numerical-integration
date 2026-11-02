"""DETEST (Hull et al. 1972, Enright & Pryce 1987): global error vs tolerance.

DETEST runs every problem at tolerances 1e-3, 1e-6 and 1e-9 and compares the
end-point error with the tolerance asked for. Adaptive RK methods control the
*local* error, so the global error over t in [0, 20] is some modest multiple
of the tolerance. Measured worst cases over all 25 problems: about 130x
(RK23), 1200x (RK45 at 1e-3, the Kepler D-problems) and 75x (DOP853). The
bounds below leave headroom over those.
"""

import numpy as np
import pytest

from fast_numerical_integration.integrators import solve_ivp
from fast_numerical_integration.problems import catalog

TOLERANCES = (1e-3, 1e-6, 1e-9)
METHODS = ("RK23", "RK45", "DOP853")
GLOBAL_ERROR_FACTOR = {"RK23": 1e3, "RK45": 1e4, "DOP853": 1e3}

DETEST = [pytest.param(p, id=p.name) for p in catalog.detest()]


def _solve(p, method, rtol, **kw):
    return solve_ivp(p.native(), p.t_span, p.y0, method=method, rtol=rtol, atol=rtol * 1e-3, **kw)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("problem", DETEST)
def test_global_error_bounded_by_tolerance(problem, method):
    for rtol in TOLERANCES:
        sol = _solve(problem, method, rtol)
        assert sol.success
        err = catalog.end_error(problem, sol.y[:, -1])
        assert err < GLOBAL_ERROR_FACTOR[method] * rtol, f"rtol={rtol:g}: error {err:.2e}"


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("problem", DETEST)
def test_tighter_tolerance_is_more_accurate_and_costs_more(problem, method):
    errs, nfevs = [], []
    for rtol in TOLERANCES:
        sol = _solve(problem, method, rtol)
        errs.append(catalog.end_error(problem, sol.y[:, -1]))
        nfevs.append(sol.nfev)
    # Each 1000x tightening must buy at least ~10x accuracy (until round-off).
    for coarse, fine in zip(errs, errs[1:]):
        assert fine < max(coarse / 10.0, 1e-12)
    assert nfevs[0] <= nfevs[1] <= nfevs[2]


@pytest.mark.parametrize("problem", DETEST)
def test_published_reference_matches_tight_solution(problem):
    """Checks the DETEST data itself (``y_end_ref`` transcribed from TEST_ODE)."""
    if problem.y_end_ref is None:
        pytest.skip("no published end value")
    tight = solve_ivp(problem.native(), problem.t_span, problem.y0, method="DOP853",
                      rtol=1e-13, atol=1e-15)
    scale = catalog.solution_scale(problem)
    np.testing.assert_allclose(tight.y[:, -1], problem.y_end_ref, rtol=0, atol=1e-10 * scale)
    if problem.exact is not None:
        exact = problem.exact(np.array([problem.t_span[1]]))[:, 0]
        np.testing.assert_allclose(problem.y_end_ref, exact, rtol=0, atol=1e-13 * scale)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("problem", [p for p in DETEST if p.values[0].exact is not None])
def test_whole_trajectory_against_exact_solution(problem, method):
    """Not just the end point: every accepted step and the dense output.

    Mid-trajectory errors can be far larger than end-point errors. On the
    Kepler D-problems a small phase error becomes a large state error at each
    periapsis passage, where the body moves fastest. D5 (e = 0.9) peaks at
    about 4e4 * rtol for RK23 while ending at about 3e2 * rtol, which is why
    DETEST scores end points. The bound here scales with the trajectory's
    largest magnitude and catches real breakage (O(1) errors), not tuning.
    """
    rtol = 1e-8
    sol = _solve(problem, method, rtol, dense_output=True)
    tq = np.linspace(*problem.t_span, 997)
    exact_q = problem.exact(tq)
    scale = max(np.max(np.abs(exact_q)), 1e-300)
    bound = 1e5 * rtol * scale
    assert np.max(np.abs(sol.y - problem.exact(sol.t))) < bound
    assert np.max(np.abs(sol.sol(tq) - exact_q)) < bound
