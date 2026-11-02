"""Parity with SciPy across the whole problem catalog.

Two oracles:

1. **Stock SciPy**, compared to accuracy. RHS evaluation counts agree to a
   few percent and end states to within tolerance. They are often exactly
   equal: always for RK23/RK45 on x86-64 Linux. But exact equality isn't a
   portable property. Near the noise floor, or when the controller rides a
   stability boundary, rounding differences flip accept/reject decisions.
   SciPy's rounding depends on the platform (NumPy's SIMD kernels, the C
   runtime's ``pow``), and so does its count. CI saw RK45 on C2 change on
   Linux aarch64. Even on one machine, a 1-ulp change to ``y0`` or the
   compiled RHS in place of the NumPy one moves SciPy's own DOP853 count.

2. **SciPy with our summation order** (the decisive check). We patch SciPy's
   ``rk_step``, error norms and RMS norm to sum in the same order as the
   Cython engine. The algorithm is otherwise SciPy's own code. SciPy then
   produces a *bit-identical* trajectory (every step time and state) for all
   three methods. This shows the only difference between the engines is
   floating-point summation order, not algorithm.

The bit-identity check assumes no fused multiply-add contraction in the
extension (true for the default x86-64 build flags).
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp as scipy_solve_ivp

from fast_numerical_integration.integrators import solve_ivp
from fast_numerical_integration.problems import catalog
from fast_numerical_integration.testing import scipy_with_our_summation_order

METHODS = ("RK23", "RK45", "DOP853")
SLOW_PROBLEMS = {"kepler_long", "outer_solar_system", "figure_eight_long", "arenstorf_long",
                 "vdp_mu1000", "linear_stiff"}


def _problem_params():
    params = []
    for p in catalog.all_problems():
        marks = [pytest.mark.slow] if p.name in SLOW_PROBLEMS else []
        params.append(pytest.param(p, id=p.name, marks=marks))
    return params


def _tolerances(method):
    return [1e-6] if method == "RK23" else [1e-6, 1e-10]


# --- tests -------------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("problem", _problem_params())
def test_bit_identical_to_scipy_with_matched_summation_order(problem, method):
    nat = problem.native()
    for rtol in _tolerances(method):
        kw = dict(method=method, rtol=rtol, atol=rtol * 1e-3)
        with scipy_with_our_summation_order():
            ref = scipy_solve_ivp(nat, problem.t_span, problem.y0, **kw)
        sol = solve_ivp(nat, problem.t_span, problem.y0, **kw)
        assert sol.status == ref.status
        assert sol.nfev == ref.nfev
        np.testing.assert_array_equal(sol.t, ref.t)
        np.testing.assert_array_equal(sol.y, ref.y)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("problem", _problem_params())
def test_matches_stock_scipy(problem, method):
    for rtol in _tolerances(method):
        kw = dict(method=method, rtol=rtol, atol=rtol * 1e-3)
        with warnings.catch_warnings():
            # Rejected trial steps on stiff problems may overflow in the NumPy RHS.
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = scipy_solve_ivp(problem.rhs, problem.t_span, problem.y0, **kw)
        sol = solve_ivp(problem.native(), problem.t_span, problem.y0, **kw)
        assert sol.status == ref.status

        # Close, not exact: see the module docstring. Exact agreement is the
        # job of test_bit_identical_to_scipy_with_matched_summation_order.
        assert sol.nfev == pytest.approx(ref.nfev, rel=0.1)

        if problem.expect_failure:
            assert sol.t[-1] == pytest.approx(ref.t[-1], rel=1e-6)
            continue
        if problem.name in ("arenstorf_long", "lorenz"):
            continue  # unstable / chaotic: end states legitimately diverge
        scale = max(np.max(np.abs(ref.y[:, -1])), 1.0)
        bound = 1e3 * rtol * scale
        diff = np.max(np.abs(sol.y[:, -1] - ref.y[:, -1]))
        if diff > bound:
            # Over long integrations both engines accumulate global error of the
            # same size along slightly different step sequences, so they drift
            # apart by about that error. Require instead that we are no less
            # accurate than SciPy against the reference solution.
            truth = catalog.reference_end(problem)
            err_ours = np.max(np.abs(sol.y[:, -1] - truth))
            err_scipy = np.max(np.abs(ref.y[:, -1] - truth))
            assert err_ours <= 2.0 * err_scipy + bound, (err_ours, err_scipy)


@pytest.mark.parametrize("problem", _problem_params())
def test_native_rhs_matches_numpy_rhs(problem):
    t1 = problem.t_span[1] if not problem.expect_failure else 0.9
    ts = np.linspace(problem.t_span[0], t1, 9)
    ys = solve_ivp(problem.native(), (problem.t_span[0], t1), problem.y0, method="DOP853",
                   rtol=1e-8, atol=1e-10, t_eval=ts).y
    nat = problem.native()
    for k, t in enumerate(ts):
        f_np = problem.rhs(t, ys[:, k])
        np.testing.assert_allclose(nat(t, ys[:, k]), f_np, rtol=1e-13,
                                   atol=1e-13 * max(1.0, np.max(np.abs(f_np))))
