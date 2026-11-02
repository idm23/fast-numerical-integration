"""Tests for the Cython RK23 / RK45 / DOP853 integrators.

The main oracle is :func:`scipy.integrate.solve_ivp`. The Cython engine copies
SciPy's algorithm exactly, so it must accept the same number of steps and make
the same number of RHS evaluations.

Step *times* are compared only loosely. The embedded error estimate
``sum_j E_j K_j`` has ``sum_j E_j = 0``, so it is a cancellation-heavy
difference. A different floating-point summation order (our loops vs. BLAS)
moves it far more than machine epsilon, and the controller passes that change on
to ``h``. The final state at the shared endpoint is compared tightly.

Exact comparisons (RHS evaluation counts, step sequences) are made against
SciPy run with our summation order (``testing.matched_solve_ivp``), which is
bit-identical to our engine on every platform. Stock SciPy's rounding varies
with the platform's SIMD kernels and C runtime, so its exact counts can move
in rounding-sensitive regimes (seen on Linux aarch64, macOS and Windows CI).
"""

import numpy as np
import pytest

from fast_numerical_integration import integrators, tableaus
from fast_numerical_integration.integrators import OdeSystem, solve_ivp
from fast_numerical_integration.nbody import NBodyGravity
from fast_numerical_integration.three_body import figure_eight_system
from fast_numerical_integration.testing import matched_solve_ivp

METHODS = ("RK23", "RK45", "DOP853")
TOLERANCES = [(1e-3, 1e-6), (1e-8, 1e-10), (1e-11, 1e-13)]


def _decay(t, y, k):
    return -k * y


def test_vendored_tableaus_match_scipy():
    import scipy.integrate as si

    for m in ("RK23", "RK45"):
        cls = getattr(si, m)
        for name in "ABCE":
            assert np.array_equal(getattr(tableaus, f"{m}_{name}"), getattr(cls, name))
    assert np.array_equal(tableaus.DOP853_A[:12, :12], si.DOP853.A)
    assert np.array_equal(tableaus.DOP853_B, si.DOP853.B)
    assert np.array_equal(tableaus.DOP853_C[:12], si.DOP853.C)
    assert np.array_equal(tableaus.DOP853_E3, si.DOP853.E3)
    assert np.array_equal(tableaus.DOP853_E5, si.DOP853.E5)
    assert np.array_equal(tableaus.DOP853_D, si.DOP853.D)


def test_native_rhs_matches_numpy_rhs():
    system = figure_eight_system()
    native = system.native_rhs()
    rng = np.random.default_rng(0)
    for _ in range(10):
        y = system.y0 + 0.1 * rng.standard_normal(system.y0.shape)
        np.testing.assert_allclose(native(0.0, y), system.rhs(0.0, y), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("rtol,atol", TOLERANCES)
@pytest.mark.parametrize("native", [True, False], ids=["native", "pyrhs"])
def test_matches_scipy_step_for_step(method, rtol, atol, native):
    system = figure_eight_system()
    t_span = (0.0, system.period)
    ref = matched_solve_ivp(system.rhs, t_span, system.y0, method=method, rtol=rtol, atol=atol)
    fun = system.native_rhs() if native else system.rhs
    sol = solve_ivp(fun, t_span, system.y0, method=method, rtol=rtol, atol=atol)

    assert sol.success and sol.status == 0
    assert sol.nfev == ref.nfev
    assert sol.t.shape == ref.t.shape
    assert sol.y.shape == ref.y.shape
    np.testing.assert_allclose(sol.t, ref.t, rtol=1e-4)
    assert sol.t[-1] == ref.t[-1] == system.period
    np.testing.assert_allclose(sol.y[:, -1], ref.y[:, -1], rtol=0, atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_exponential_decay_accuracy(method):
    k = 1.7
    y0 = np.array([1.0, -2.0, 0.5])
    sol = solve_ivp(_decay, (0.0, 3.0), y0, method=method, rtol=1e-10, atol=1e-12, args=(k,))
    assert sol.success
    np.testing.assert_allclose(sol.y[:, -1], y0 * np.exp(-k * 3.0), rtol=1e-7)


@pytest.mark.parametrize("method", METHODS)
def test_backward_integration_matches_scipy(method):
    y0 = np.array([1.0, 2.0])
    ref = matched_solve_ivp(_decay, (2.0, -1.0), y0, method=method, rtol=1e-8, atol=1e-10, args=(0.5,))
    sol = solve_ivp(_decay, (2.0, -1.0), y0, method=method, rtol=1e-8, atol=1e-10, args=(0.5,))
    assert sol.success
    assert sol.nfev == ref.nfev
    assert sol.t[-1] == -1.0
    # Same steps; values agree to well inside the requested rtol=1e-8. Exact
    # agreement isn't expected: stock scipy's rounding depends on the CPU's
    # BLAS kernel (on some CI runners the step values differ by ~1e-9).
    np.testing.assert_allclose(sol.y, ref.y, rtol=1e-7)


@pytest.mark.parametrize("method", METHODS)
def test_vector_atol_first_step_and_max_step(method):
    system = figure_eight_system()
    atol = np.linspace(1e-12, 1e-9, system.y0.size)
    kw = dict(method=method, rtol=1e-8, atol=atol, first_step=1e-3, max_step=0.05)
    t_span = (0.0, system.period)
    ref = matched_solve_ivp(system.rhs, t_span, system.y0, **kw)
    sol = solve_ivp(system.native_rhs(), t_span, system.y0, **kw)
    assert sol.nfev == ref.nfev
    assert sol.t.shape == ref.t.shape
    assert np.all(np.diff(sol.t) <= 0.05 * (1 + 1e-12))
    np.testing.assert_allclose(sol.y[:, -1], ref.y[:, -1], rtol=0, atol=1e-9)


def test_zero_length_interval_matches_scipy():
    y0 = np.array([1.0, 2.0])
    ref = matched_solve_ivp(_decay, (1.0, 1.0), y0, args=(1.0,))
    sol = solve_ivp(_decay, (1.0, 1.0), y0, args=(1.0,))
    assert sol.success
    np.testing.assert_array_equal(sol.t, ref.t)
    np.testing.assert_array_equal(sol.y, ref.y)


def test_stepper_api():
    system = figure_eight_system()
    solver = integrators.DOP853(system.native_rhs(), 0.0, system.y0, system.period, rtol=1e-8, atol=1e-10)
    assert solver.status == "running"
    steps = 0
    while solver.status == "running":
        solver.step()
        steps += 1
    assert solver.status == "finished"
    assert solver.t == system.period
    assert solver.n_accepted == steps
    with pytest.raises(RuntimeError):
        solver.step()


def test_python_rhs_exceptions_propagate():
    def bad(t, y):
        raise ZeroDivisionError("boom")

    with pytest.raises(ZeroDivisionError, match="boom"):
        solve_ivp(bad, (0.0, 1.0), [1.0])


def test_python_rhs_wrong_size_rejected():
    with pytest.raises(ValueError, match="expected 2"):
        solve_ivp(lambda t, y: np.zeros(3), (0.0, 1.0), [1.0, 2.0])


def test_base_system_is_abstract():
    with pytest.raises(NotImplementedError):
        solve_ivp(OdeSystem(), (0.0, 1.0), [1.0])


def test_native_system_size_mismatch():
    with pytest.raises(ValueError, match="expects 12 states"):
        solve_ivp(NBodyGravity(np.ones(3)), (0.0, 1.0), np.zeros(4))


def test_invalid_arguments():
    with pytest.raises(ValueError, match="method"):
        solve_ivp(_decay, (0.0, 1.0), [1.0], method="Radau", args=(1.0,))
    with pytest.raises(ValueError, match="max_step"):
        solve_ivp(_decay, (0.0, 1.0), [1.0], max_step=0.0, args=(1.0,))
    with pytest.raises(ValueError, match="first_step"):
        solve_ivp(_decay, (0.0, 1.0), [1.0], first_step=2.0, args=(1.0,))
    with pytest.raises(ValueError, match="atol"):
        solve_ivp(_decay, (0.0, 1.0), [1.0, 2.0], atol=[1e-6, 1e-6, 1e-6], args=(1.0,))
