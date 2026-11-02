"""Port of SciPy's own ``solve_ivp`` tests to the Cython engine.

Source: ``scipy/integrate/_ivp/tests/test_ivp.py`` and ``test_rk.py`` (SciPy
1.17). Only tests that apply to RK23 / RK45 / DOP853 are ported; tests of
Radau, BDF, LSODA, Jacobians, sparsity and complex-valued problems are left
out.

Each scenario runs via :func:`both`. Status, message and events must agree
with stock SciPy, and the RHS evaluation count and output shape must agree
exactly with SciPy run in our summation order. SciPy's original assertions
are then applied to our result.

Two deliberate adaptations, both noted where they occur:

- With ``t_eval`` and no output points, we return empty arrays where SciPy
  returns empty lists.
- Tests that call private SciPy helpers (``select_initial_step``,
  ``ConstantDenseOutput``, ``_estimate_error``) use the equivalent public
  behaviour of our engine instead.
"""

import warnings
from itertools import product

import numpy as np
import pytest
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_array_less,
    assert_equal,
    assert_no_warnings,
)
from pytest import raises as assert_raises
from scipy.integrate import solve_ivp as scipy_solve_ivp

from fast_numerical_integration import tableaus
from fast_numerical_integration.integrators import (
    DOP853,
    RK23,
    RK45,
    OdeSolution,
    solve_ivp,
)
from fast_numerical_integration.testing import matched_solve_ivp

METHODS = ["RK23", "RK45", "DOP853"]
CLASSES = [RK23, RK45, DOP853]


# --- helpers (verbatim from scipy's test_ivp.py) ------------------------------

def fun_zero(t, y):
    return np.zeros_like(y)


def fun_linear(t, y):
    return np.array([-y[0] - 5 * y[1], y[0] + y[1]])


def sol_linear(t):
    return np.vstack((-5 * np.sin(2 * t), 2 * np.cos(2 * t) + np.sin(2 * t)))


def fun_rational(t, y):
    return np.array([y[1] / t, y[1] * (y[0] + 2 * y[1] - 1) / (t * (y[0] - 1))])


def fun_rational_vectorized(t, y):
    return np.vstack((y[1] / t, y[1] * (y[0] + 2 * y[1] - 1) / (t * (y[0] - 1))))


def jac_rational(t, y):
    return np.array([
        [0, 1 / t],
        [-2 * y[1] ** 2 / (t * (y[0] - 1) ** 2), (y[0] + 4 * y[1] - 1) / (t * (y[0] - 1))],
    ])


def sol_rational(t):
    return np.asarray((t / (t + 10), 10 * t / (t + 10) ** 2))


def compute_error(y, y_true, rtol, atol):
    e = (y - y_true) / (atol + rtol * np.abs(y_true))
    return np.linalg.norm(e, axis=0) / np.sqrt(e.shape[0])


def both(*args, **kwargs):
    """Run our engine, stock scipy, and scipy with our summation order.

    Status, message, event counts and times must agree with stock scipy. RHS
    evaluation count and output shape must agree *exactly* with the
    matched-order run, which is bit-identical to ours on every platform;
    stock scipy's counts can move with platform rounding. Returns
    (ours, stock scipy's).
    """
    ref = scipy_solve_ivp(*args, **kwargs)
    matched = matched_solve_ivp(*args, **kwargs)
    res = solve_ivp(*args, **kwargs)
    assert res.status == ref.status
    assert res.message == ref.message
    assert res.nfev == matched.nfev
    assert res.nfev == pytest.approx(ref.nfev, rel=0.1, abs=12)
    assert np.shape(res.t) == np.shape(matched.t) or (np.size(res.t) == 0 and len(matched.t) == 0)
    assert np.shape(res.y) == np.shape(matched.y) or (np.size(res.y) == 0 and len(matched.y) == 0)
    if ref.t_events is None:
        assert res.t_events is None
    else:
        assert [te.shape for te in res.t_events] == [te.shape for te in ref.t_events]
        for te, te_ref in zip(res.t_events, ref.t_events):
            assert_allclose(te, te_ref, rtol=1e-6, atol=1e-9)
    return res, ref


# --- test_rk.py -----------------------------------------------------------------

@pytest.mark.parametrize("name", ["RK23", "RK45"])
def test_coefficient_properties(name):
    B = getattr(tableaus, f"{name}_B")
    A = getattr(tableaus, f"{name}_A")
    C = getattr(tableaus, f"{name}_C")
    assert_allclose(np.sum(B), 1, rtol=1e-15)
    assert_allclose(np.sum(A, axis=1), C, rtol=1e-14)


def test_coefficient_properties_dop853():
    assert_allclose(np.sum(tableaus.DOP853_B), 1, rtol=1e-15)
    assert_allclose(np.sum(tableaus.DOP853_A, axis=1), tableaus.DOP853_C, rtol=1e-14)


@pytest.mark.parametrize("solver_class", CLASSES)
def test_error_estimation(solver_class):
    # Adapted: scipy inspects the private ``_estimate_error``. We check the same
    # property through behaviour: with first_step=0.2 on y' = y, the first step
    # is accepted with an actual error well inside the tolerance it reports.
    step = 0.2
    solver = solver_class(lambda t, y: y, 0, [1], 1, first_step=step)
    solver.step()
    assert solver.t == pytest.approx(step)
    assert abs(solver.y[0] - np.exp(step)) < 1e-3 * np.exp(step)


# --- test_ivp.py ------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_duplicate_timestamps(method):
    def upward_cannon(t, y):
        return [y[1], -9.80665]

    def hit_ground(t, y):
        return y[0]

    hit_ground.terminal = True
    hit_ground.direction = -1

    sol, _ = both(upward_cannon, [0, np.inf], [0, 0.01], method=method,
                  max_step=0.05 * 0.001 / 9.80665, events=hit_ground, dense_output=True)
    if method == "DOP853":
        # scipy runs this test with RK45 only. sol(0.01) extrapolates ~2000 step
        # lengths past the last step: RK45's quartic reproduces this quadratic
        # trajectory exactly, but DOP853's degree-7 form amplifies round-off
        # (scipy's own DOP853 returns garbage there as well). Check inside the
        # interval instead.
        t_mid = 0.5 * sol.t_events[0][0]
        assert_allclose(sol.sol(t_mid), [0.01 * t_mid - 0.5 * 9.80665 * t_mid**2,
                                         0.01 - 9.80665 * t_mid], rtol=1e-6, atol=1e-10)
    else:
        assert_allclose(sol.sol(0.01), np.asarray([-0.00039033, -0.08806632]), rtol=1e-5, atol=1e-8)
    assert_allclose(sol.t_events, np.asarray([[0.00203943]]), rtol=1e-5, atol=1e-8)
    assert_allclose(sol.y_events, [np.asarray([[0.0, -0.01]])], atol=1e-9)
    assert sol.success
    assert_equal(sol.status, 1)


def test_integration():
    rtol = 1e-3
    atol = 1e-6
    y0 = [1 / 3, 2 / 9]

    for vectorized, method, t_span in product([False, True], METHODS, [[5, 9], [5, 1]]):
        fun = fun_rational_vectorized if vectorized else fun_rational
        res, _ = both(fun, t_span, y0, rtol=rtol, atol=atol, method=method,
                      dense_output=True, vectorized=vectorized)

        assert_equal(res.t[0], t_span[0])
        assert_(res.t_events is None)
        assert_(res.y_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)
        if method == "DOP853":
            assert_(res.nfev < 50)
        else:
            assert_(res.nfev < 40)
        assert_equal(res.njev, 0)
        assert_equal(res.nlu, 0)

        y_true = sol_rational(res.t)
        e = compute_error(res.y, y_true, rtol, atol)
        assert_(np.all(e < 5))

        tc = np.linspace(*t_span)
        e = compute_error(res.sol(tc), sol_rational(tc), rtol, atol)
        assert_(np.all(e < 5))

        tc = (t_span[0] + t_span[-1]) / 2
        e = compute_error(res.sol(tc), sol_rational(tc), rtol, atol)
        assert_(np.all(e < 5))

        assert_allclose(res.sol(res.t), res.y, rtol=1e-15, atol=1e-15)


def test_integration_jac_warns_like_scipy():
    # scipy's test passes jac= to every method; for RK methods it only warns.
    msg = "The following arguments have no effect for a chosen solver: `jac`"
    for method in METHODS:
        with pytest.warns(UserWarning, match=msg):
            scipy_solve_ivp(fun_rational, [5, 9], [1 / 3, 2 / 9], method=method, jac=jac_rational)
        with pytest.warns(UserWarning, match=msg):
            solve_ivp(fun_rational, [5, 9], [1 / 3, 2 / 9], method=method, jac=jac_rational)


def _rational_events():
    def event_rational_1(t, y):
        return y[0] - y[1] ** 0.7

    def event_rational_2(t, y):
        return y[1] ** 0.6 - y[0]

    def event_rational_3(t, y):
        return t - 7.4

    event_rational_3.terminal = True
    return event_rational_1, event_rational_2, event_rational_3


@pytest.mark.parametrize("method", METHODS)
def test_events_forward(method):
    event_rational_1, event_rational_2, event_rational_3 = _rational_events()

    res, _ = both(fun_rational, [5, 8], [1 / 3, 2 / 9], method=method,
                  events=(event_rational_1, event_rational_2))
    assert_equal(res.status, 0)
    assert_equal(res.t_events[0].size, 1)
    assert_equal(res.t_events[1].size, 1)
    assert_(5.3 < res.t_events[0][0] < 5.7)
    assert_(7.3 < res.t_events[1][0] < 7.7)
    assert_equal(res.y_events[0].shape, (1, 2))
    assert_equal(res.y_events[1].shape, (1, 2))
    assert np.isclose(event_rational_1(res.t_events[0][0], res.y_events[0][0]), 0)
    assert np.isclose(event_rational_2(res.t_events[1][0], res.y_events[1][0]), 0)

    event_rational_1.direction = 1
    event_rational_2.direction = 1
    res, _ = both(fun_rational, [5, 8], [1 / 3, 2 / 9], method=method,
                  events=(event_rational_1, event_rational_2))
    assert_equal(res.status, 0)
    assert_equal(res.t_events[0].size, 1)
    assert_equal(res.t_events[1].size, 0)
    assert_(5.3 < res.t_events[0][0] < 5.7)
    assert_equal(res.y_events[0].shape, (1, 2))
    assert_equal(res.y_events[1].shape, (0,))
    assert np.isclose(event_rational_1(res.t_events[0][0], res.y_events[0][0]), 0)

    event_rational_1.direction = -1
    event_rational_2.direction = -1
    res, _ = both(fun_rational, [5, 8], [1 / 3, 2 / 9], method=method,
                  events=(event_rational_1, event_rational_2))
    assert_equal(res.status, 0)
    assert_equal(res.t_events[0].size, 0)
    assert_equal(res.t_events[1].size, 1)
    assert_(7.3 < res.t_events[1][0] < 7.7)
    assert_equal(res.y_events[0].shape, (0,))
    assert_equal(res.y_events[1].shape, (1, 2))
    assert np.isclose(event_rational_2(res.t_events[1][0], res.y_events[1][0]), 0)

    event_rational_1.direction = 0
    event_rational_2.direction = 0
    res, _ = both(fun_rational, [5, 8], [1 / 3, 2 / 9], method=method,
                  events=(event_rational_1, event_rational_2, event_rational_3),
                  dense_output=True)
    assert_equal(res.status, 1)
    assert_equal(res.t_events[0].size, 1)
    assert_equal(res.t_events[1].size, 0)
    assert_equal(res.t_events[2].size, 1)
    assert_(5.3 < res.t_events[0][0] < 5.7)
    assert_(7.3 < res.t_events[2][0] < 7.5)
    assert_equal(res.y_events[0].shape, (1, 2))
    assert_equal(res.y_events[1].shape, (0,))
    assert_equal(res.y_events[2].shape, (1, 2))
    assert np.isclose(event_rational_1(res.t_events[0][0], res.y_events[0][0]), 0)
    assert np.isclose(event_rational_3(res.t_events[2][0], res.y_events[2][0]), 0)

    res, _ = both(fun_rational, [5, 8], [1 / 3, 2 / 9], method=method,
                  events=event_rational_1, dense_output=True)
    assert_equal(res.status, 0)
    assert_equal(res.t_events[0].size, 1)
    assert_(5.3 < res.t_events[0][0] < 5.7)
    assert_equal(res.y_events[0].shape, (1, 2))
    assert np.isclose(event_rational_1(res.t_events[0][0], res.y_events[0][0]), 0)

    # Termination by event doesn't break interpolants.
    tc = np.linspace(res.t[0], res.t[-1])
    e = compute_error(res.sol(tc), sol_rational(tc), 1e-3, 1e-6)
    assert_(np.all(e < 5))
    assert np.allclose(sol_rational(res.t_events[0][0]), res.y_events[0][0], rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_events_backward(method):
    event_rational_1, event_rational_2, event_rational_3 = _rational_events()

    res, _ = both(fun_rational, [8, 5], [4 / 9, 20 / 81], method=method,
                  events=(event_rational_1, event_rational_2))
    assert_equal(res.status, 0)
    assert_equal(res.t_events[0].size, 1)
    assert_equal(res.t_events[1].size, 1)
    assert_(5.3 < res.t_events[0][0] < 5.7)
    assert_(7.3 < res.t_events[1][0] < 7.7)
    assert_equal(res.y_events[0].shape, (1, 2))
    assert_equal(res.y_events[1].shape, (1, 2))
    assert np.isclose(event_rational_1(res.t_events[0][0], res.y_events[0][0]), 0)
    assert np.isclose(event_rational_2(res.t_events[1][0], res.y_events[1][0]), 0)

    event_rational_1.direction = -1
    event_rational_2.direction = -1
    res, _ = both(fun_rational, [8, 5], [4 / 9, 20 / 81], method=method,
                  events=(event_rational_1, event_rational_2))
    assert_equal(res.status, 0)
    assert_equal(res.t_events[0].size, 1)
    assert_equal(res.t_events[1].size, 0)
    assert_(5.3 < res.t_events[0][0] < 5.7)
    assert_equal(res.y_events[0].shape, (1, 2))
    assert_equal(res.y_events[1].shape, (0,))
    assert np.isclose(event_rational_1(res.t_events[0][0], res.y_events[0][0]), 0)

    event_rational_1.direction = 1
    event_rational_2.direction = 1
    res, _ = both(fun_rational, [8, 5], [4 / 9, 20 / 81], method=method,
                  events=(event_rational_1, event_rational_2))
    assert_equal(res.status, 0)
    assert_equal(res.t_events[0].size, 0)
    assert_equal(res.t_events[1].size, 1)
    assert_(7.3 < res.t_events[1][0] < 7.7)
    assert_equal(res.y_events[0].shape, (0,))
    assert_equal(res.y_events[1].shape, (1, 2))
    assert np.isclose(event_rational_2(res.t_events[1][0], res.y_events[1][0]), 0)

    event_rational_1.direction = 0
    event_rational_2.direction = 0
    res, _ = both(fun_rational, [8, 5], [4 / 9, 20 / 81], method=method,
                  events=(event_rational_1, event_rational_2, event_rational_3),
                  dense_output=True)
    assert_equal(res.status, 1)
    assert_equal(res.t_events[0].size, 0)
    assert_equal(res.t_events[1].size, 1)
    assert_equal(res.t_events[2].size, 1)
    assert_(7.3 < res.t_events[1][0] < 7.7)
    assert_(7.3 < res.t_events[2][0] < 7.5)
    assert_equal(res.y_events[0].shape, (0,))
    assert_equal(res.y_events[1].shape, (1, 2))
    assert_equal(res.y_events[2].shape, (1, 2))
    assert np.isclose(event_rational_2(res.t_events[1][0], res.y_events[1][0]), 0)
    assert np.isclose(event_rational_3(res.t_events[2][0], res.y_events[2][0]), 0)

    tc = np.linspace(res.t[-1], res.t[0])
    e = compute_error(res.sol(tc), sol_rational(tc), 1e-3, 1e-6)
    assert_(np.all(e < 5))
    assert np.allclose(sol_rational(res.t_events[1][0]), res.y_events[1][0], rtol=1e-3, atol=1e-6)
    assert np.allclose(sol_rational(res.t_events[2][0]), res.y_events[2][0], rtol=1e-3, atol=1e-6)


def _get_harmonic_oscillator():
    def f(t, y):
        return [y[1], -y[0]]

    def event(t, y):
        return y[0]

    return f, event


@pytest.mark.parametrize("n_events", [3, 4])
def test_event_terminal_integer(n_events):
    f, event = _get_harmonic_oscillator()
    event.terminal = n_events
    res, _ = both(f, (0, 100), [1, 0], events=event)
    assert len(res.t_events[0]) == n_events
    assert len(res.y_events[0]) == n_events
    assert_allclose(res.y_events[0][:, 0], 0, atol=1e-14)


def test_event_terminal_iv():
    f, event = _get_harmonic_oscillator()
    args = (f, (0, 100), [1, 0])

    event.terminal = None
    res, _ = both(*args, events=event)
    event.terminal = 0
    ref, _ = both(*args, events=event)
    assert_allclose(res.t_events, ref.t_events)

    message = "The `terminal` attribute..."
    event.terminal = -1
    with pytest.raises(ValueError, match=message):
        solve_ivp(*args, events=event)
    event.terminal = 3.5
    with pytest.raises(ValueError, match=message):
        solve_ivp(*args, events=event)


@pytest.mark.parametrize("method", CLASSES)
def test_max_step(method):
    rtol = 1e-3
    atol = 1e-6
    y0 = [1 / 3, 2 / 9]
    for t_span in ([5, 9], [5, 1]):
        res, _ = both(fun_rational, t_span, y0, rtol=rtol, max_step=0.5, atol=atol,
                      method=method.__name__, dense_output=True)
        assert_equal(res.t[0], t_span[0])
        assert_equal(res.t[-1], t_span[-1])
        assert_(np.all(np.abs(np.diff(res.t)) <= 0.5 + 1e-15))
        assert_(res.t_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)

        e = compute_error(res.y, sol_rational(res.t), rtol, atol)
        assert_(np.all(e < 5))
        tc = np.linspace(*t_span)
        e = compute_error(res.sol(tc), sol_rational(tc), rtol, atol)
        assert_(np.all(e < 5))
        assert_allclose(res.sol(res.t), res.y, rtol=1e-15, atol=1e-15)

        assert_raises(ValueError, method, fun_rational, t_span[0], y0, t_span[1], max_step=-1)

        solver = method(fun_rational, t_span[0], y0, t_span[1], rtol=rtol, atol=atol, max_step=1e-20)
        message = solver.step()
        message = solver.step()  # First step succeeds but second step fails.
        assert_equal(solver.status, "failed")
        assert_("step size is less" in message)
        assert_raises(RuntimeError, solver.step)


@pytest.mark.parametrize("method", CLASSES)
def test_first_step(method):
    rtol = 1e-3
    atol = 1e-6
    y0 = [1 / 3, 2 / 9]
    first_step = 0.1
    for t_span in ([5, 9], [5, 1]):
        res, _ = both(fun_rational, t_span, y0, rtol=rtol, max_step=0.5, atol=atol,
                      method=method.__name__, dense_output=True, first_step=first_step)
        assert_equal(res.t[0], t_span[0])
        assert_equal(res.t[-1], t_span[-1])
        assert_allclose(first_step, np.abs(res.t[1] - 5))
        assert_(res.t_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)

        e = compute_error(res.y, sol_rational(res.t), rtol, atol)
        assert_(np.all(e < 5))
        tc = np.linspace(*t_span)
        e = compute_error(res.sol(tc), sol_rational(tc), rtol, atol)
        assert_(np.all(e < 5))
        assert_allclose(res.sol(res.t), res.y, rtol=1e-15, atol=1e-15)

        assert_raises(ValueError, method, fun_rational, t_span[0], y0, t_span[1], first_step=-1)
        assert_raises(ValueError, method, fun_rational, t_span[0], y0, t_span[1], first_step=5)


def test_t_eval():
    rtol = 1e-3
    atol = 1e-6
    y0 = [1 / 3, 2 / 9]
    for t_span in ([5, 9], [5, 1]):
        t_eval = np.linspace(t_span[0], t_span[1], 10)
        res, _ = both(fun_rational, t_span, y0, rtol=rtol, atol=atol, t_eval=t_eval)
        assert_equal(res.t, t_eval)
        assert_(res.t_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)
        e = compute_error(res.y, sol_rational(res.t), rtol, atol)
        assert_(np.all(e < 5))

    for t_span, t_eval in [([5, 9], [5, 5.01, 7, 8, 8.01, 9]),
                           ([5, 1], [5, 4.99, 3, 1.5, 1.1, 1.01, 1]),
                           ([5, 9], [5.01, 7, 8, 8.01]),
                           ([5, 1], [4.99, 3, 1.5, 1.1, 1.01])]:
        res, _ = both(fun_rational, t_span, y0, rtol=rtol, atol=atol, t_eval=t_eval)
        assert_equal(res.t, t_eval)
        assert_(res.t_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)
        e = compute_error(res.y, sol_rational(res.t), rtol, atol)
        assert_(np.all(e < 5))

    assert_raises(ValueError, solve_ivp, fun_rational, [5, 9], y0, rtol=rtol, atol=atol, t_eval=[4, 6])


def test_t_eval_dense_output():
    rtol = 1e-3
    atol = 1e-6
    y0 = [1 / 3, 2 / 9]
    t_span = [5, 9]
    t_eval = np.linspace(t_span[0], t_span[1], 10)
    res, _ = both(fun_rational, t_span, y0, rtol=rtol, atol=atol, t_eval=t_eval)
    res_d, _ = both(fun_rational, t_span, y0, rtol=rtol, atol=atol, t_eval=t_eval, dense_output=True)
    assert_equal(res.t, t_eval)
    assert_(res.t_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)

    assert_equal(res.t, res_d.t)
    assert_equal(res.y, res_d.y)
    assert_(res_d.t_events is None)
    assert_(res_d.success)
    assert_equal(res_d.status, 0)

    e = compute_error(res.y, sol_rational(res.t), rtol, atol)
    assert_(np.all(e < 5))


@pytest.mark.parametrize("method", METHODS)
def test_t_eval_early_event(method):
    def early_event(t, y):
        return t - 7

    early_event.terminal = True

    t_eval = np.linspace(7.5, 9, 16)
    res, ref = both(fun_rational, [5, 9], [1 / 3, 2 / 9], rtol=1e-3, atol=1e-6,
                    method=method, t_eval=t_eval, events=early_event)
    assert res.success
    assert res.message == "A termination event occurred."
    assert res.status == 1
    # Adapted: scipy returns empty *lists* here (`not res.t`); we return empty arrays.
    assert not ref.t and not ref.y
    assert res.t.size == 0 and res.y.size == 0
    assert len(res.t_events) == 1
    assert res.t_events[0].size == 1
    assert res.t_events[0][0] == 7


def test_no_integration():
    for method in METHODS:
        sol, _ = both(lambda t, y: -y, [4, 4], [2, 3], method=method, dense_output=True)
        assert_equal(sol.sol(4), [2, 3])
        assert_equal(sol.sol([4, 5, 6]), [[2, 2, 2], [3, 3, 3]])


def test_no_integration_class():
    for method in CLASSES:
        solver = method(lambda t, y: -y, 0.0, [10.0, 0.0], 0.0)
        solver.step()
        assert_equal(solver.status, "finished")
        sol = solver.dense_output()
        assert_equal(sol(0.0), [10.0, 0.0])
        assert_equal(sol([0, 1, 2]), [[10, 10, 10], [0, 0, 0]])

        solver = method(lambda t, y: -y, 0.0, [], np.inf)
        solver.step()
        assert_equal(solver.status, "finished")
        sol = solver.dense_output()
        assert_equal(sol(100.0), [])
        assert_equal(sol([0, 1, 2]), np.empty((0, 3)))


def test_empty():
    def fun(t, y):
        return np.zeros((0,))

    y0 = np.zeros((0,))
    for t_span in ([0, 10], [0, np.inf]):
        for method in METHODS:
            sol = assert_no_warnings(solve_ivp, fun, t_span, y0, method=method, dense_output=True)
            assert_equal(sol.sol(10), np.zeros((0,)))
            assert_equal(sol.sol([1, 2, 3]), np.zeros((0, 3)))


def _constant_segments(ts, values):
    """OdeSolution whose segments are constant (scipy's ConstantDenseOutput)."""
    ts = np.asarray(ts, dtype=float)
    values = [np.atleast_1d(np.asarray(v, dtype=float)) for v in values]
    n = values[0].size
    ncoef = 4
    rows = [np.concatenate(([t_old, 0.0], v, np.zeros(ncoef * n))) for t_old, v in zip(ts[:-1], values)]
    return OdeSolution(ts, np.array(rows), 0, n, ncoef)


def test_ConstantDenseOutput():
    # Adapted: scipy's private ConstantDenseOutput is a zero-length segment here.
    sol = _constant_segments([0, 1], [[1, 2]])
    assert_allclose(sol(1.5), [1, 2])
    assert_allclose(sol([1, 1.5, 2]), [[1, 1, 1], [2, 2, 2]])


def test_classes():
    y0 = [1 / 3, 2 / 9]
    for cls in CLASSES:
        solver = cls(fun_rational, 5, y0, np.inf)
        assert_equal(solver.n, 2)
        assert_equal(solver.status, "running")
        assert_equal(solver.t_bound, np.inf)
        assert_equal(solver.direction, 1)
        assert_equal(solver.t, 5)
        assert_equal(solver.y, y0)
        assert_(solver.step_size is None)
        assert_(solver.nfev > 0)
        assert_(solver.njev >= 0)
        assert_equal(solver.nlu, 0)
        assert_raises(RuntimeError, solver.dense_output)
        message = solver.step()
        assert_equal(solver.status, "running")
        assert_equal(message, None)
        assert_equal(solver.n, 2)
        assert_equal(solver.t_bound, np.inf)
        assert_equal(solver.direction, 1)
        assert_(solver.t > 5)
        assert_(not np.all(np.equal(solver.y, y0)))
        assert_(solver.step_size > 0)
        assert_(solver.nfev > 0)
        assert_(solver.njev >= 0)
        assert_(solver.nlu >= 0)
        sol = solver.dense_output()
        assert_allclose(sol(5), y0, rtol=1e-15, atol=0)


def test_OdeSolution():
    sol = _constant_segments([0, 2, 5], [[-1], [1]])
    assert_equal(sol(-1), [-1])
    assert_equal(sol(1), [-1])
    assert_equal(sol(2), [-1])
    assert_equal(sol(3), [1])
    assert_equal(sol(5), [1])
    assert_equal(sol(6), [1])
    assert_equal(sol([0, 6, -2, 1.5, 4.5, 2.5, 5, 5.5, 2]),
                 np.array([[-1, 1, -1, -1, 1, 1, 1, 1, -1]]))

    sol = _constant_segments([10, 4, -3], [[-1], [1]])
    assert_equal(sol(11), [-1])
    assert_equal(sol(10), [-1])
    assert_equal(sol(5), [-1])
    assert_equal(sol(4), [-1])
    assert_equal(sol(0), [1])
    assert_equal(sol(-3), [1])
    assert_equal(sol(-4), [1])
    assert_equal(sol([12, -5, 10, -3, 6, 1, 4]), np.array([[-1, 1, -1, 1, -1, 1, -1]]))

    sol = _constant_segments([1, 1], [[10]])
    assert_equal(sol(0), [10])
    assert_equal(sol(1), [10])
    assert_equal(sol(2), [10])
    assert_equal(sol([2, 1, 0]), np.array([[10, 10, 10]]))


def test_args():
    # Adapted: scipy uses Radau + jac here; we use DOP853 at the same tolerances.
    def sys3(t, w, omega, k, zfinal):
        x, y, z = w
        return [-omega * y, omega * x, k * z * (1 - z)]

    def sys3_x0decreasing(t, w, omega, k, zfinal):
        x, y, z = w
        return x

    def sys3_y0increasing(t, w, omega, k, zfinal):
        x, y, z = w
        return y

    def sys3_zfinal(t, w, omega, k, zfinal):
        x, y, z = w
        return z - zfinal

    sys3_x0decreasing.direction = -1
    sys3_y0increasing.direction = 1
    sys3_zfinal.terminal = True

    omega = 2
    k = 4
    tfinal = 5
    zfinal = 0.99
    z0 = np.exp(-k * tfinal) / ((1 - zfinal) / zfinal + np.exp(-k * tfinal))
    w0 = [0, -1, z0]
    tend = 2 * tfinal

    sol, _ = both(sys3, [0, tend], w0,
                  events=[sys3_x0decreasing, sys3_y0increasing, sys3_zfinal],
                  dense_output=True, args=(omega, k, zfinal), method="DOP853",
                  rtol=1e-10, atol=1e-13)

    x0events_t, y0events_t, zfinalevents_t = sol.t_events
    assert_allclose(x0events_t, [0.5 * np.pi, 1.5 * np.pi])
    assert_allclose(y0events_t, [0.25 * np.pi, 1.25 * np.pi])
    assert_allclose(zfinalevents_t, [tfinal])

    t = np.linspace(0, zfinalevents_t[0], 250)
    w = sol.sol(t)
    # atol 1e-12 -> 5e-10: scipy's tolerances were set for Radau. DOP853 (in
    # scipy too: identical 2.9e-9 relative miss) is ~1.5e-10 off near the zero
    # crossings of sin/cos, where rtol is meaningless.
    assert_allclose(w[0], np.sin(omega * t), rtol=1e-9, atol=5e-10)
    assert_allclose(w[1], -np.cos(omega * t), rtol=1e-9, atol=5e-10)
    # rtol 1e-9 -> 1e-7 for the logistic component: scipy's DOP853 has the same
    # 4.05e-8 relative error here (the 1e-9 bound was set for Radau).
    assert_allclose(w[2], 1 / (((1 - z0) / z0) * np.exp(-k * t) + 1), rtol=1e-7, atol=1e-12)

    x0events = sol.sol(x0events_t)
    y0events = sol.sol(y0events_t)
    zfinalevents = sol.sol(zfinalevents_t)
    assert_allclose(x0events[0], np.zeros_like(x0events[0]), atol=5e-14)
    assert_allclose(x0events[1], np.ones_like(x0events[1]))
    assert_allclose(y0events[0], np.ones_like(y0events[0]))
    assert_allclose(y0events[1], np.zeros_like(y0events[1]), atol=5e-14)
    assert_allclose(zfinalevents[2], [zfinal])


def test_array_rtol():
    def f(t, y):
        return y[0], y[1]

    sol, _ = both(f, (0, 1), [1.0, 1.0], rtol=[1e-1, 1e-1])
    err1 = np.abs(np.linalg.norm(sol.y[:, -1] - np.exp(1)))

    with pytest.warns(UserWarning, match="At least one element..."):
        sol = solve_ivp(f, (0, 1), [1.0, 1.0], rtol=[1e-1, 1e-16])
        err2 = np.abs(np.linalg.norm(sol.y[:, -1] - np.exp(1)))

    assert err2 < err1


@pytest.mark.parametrize("method", METHODS)
def test_integration_zero_rhs(method):
    result, _ = both(fun_zero, [0, 10], np.ones(3), method=method)
    assert_(result.success)
    assert_equal(result.status, 0)
    assert_allclose(result.y, 1.0, rtol=1e-15)


def test_args_single_value():
    def fun_with_arg(t, y, a):
        return a * y

    message = "Supplied 'args' cannot be unpacked."
    with pytest.raises(TypeError, match=message):
        solve_ivp(fun_with_arg, (0, 0.1), [1], args=-1)

    sol, _ = both(fun_with_arg, (0, 0.1), [1], args=(-1,))
    assert_allclose(sol.y[0, -1], np.exp(-0.1))


@pytest.mark.parametrize("f0_fill", [np.nan, np.inf])
def test_initial_state_finiteness(f0_fill):
    msg = "All components of the initial state `y0` must be finite."
    with pytest.raises(ValueError, match=msg):
        solve_ivp(fun_zero, [0, 10], np.full(3, f0_fill))


@pytest.mark.parametrize("method", METHODS)
def test_zero_interval(method):
    def f(t, y):
        return 2 * y

    res, _ = both(f, (0.0, 0.0), np.array([1.0]), method=method)
    assert res.success
    assert_allclose(res.y[0, -1], 1.0)


@pytest.mark.parametrize("method", METHODS)
def test_tbound_respected_small_interval(method):
    SMALL = 1e-4

    def f(t, y):
        if t > SMALL:
            raise ValueError("Function was evaluated outside interval")
        return 2 * y

    res, _ = both(f, (0.0, SMALL), np.array([1]), method=method)
    assert res.success


@pytest.mark.parametrize("method", METHODS)
def test_tbound_respected_larger_interval(method):
    def V(r):
        return -11 / r + 10 * r / (0.05 + r ** 2)

    def func(t, p):
        if t < -17 or t > 2:
            raise ValueError("Function was evaluated outside interval")
        P = p[0]
        Q = p[1]
        r = np.exp(t)
        dPdr = r * Q
        dQdr = -2.0 * r * ((-0.2 - V(r)) * P + 1 / r * Q)
        return np.array([dPdr, dQdr])

    result, _ = both(func, (-17, 2), y0=np.array([1, -11]), method=method, max_step=0.03,
                     vectorized=False, t_eval=None, atol=1e-8, rtol=1e-5)
    assert result.success


@pytest.mark.parametrize("method", METHODS)
def test_tbound_respected_oscillator(method):
    def reactions_func(t, y):
        if t > 205:
            raise ValueError("Called outside interval")
        return np.array([1.73307544e-02, 6.49376470e-06, 0.00000000e00, 0.00000000e00])

    init_state = np.array([134.08298555, 138.82348612, 100.0, 0.0])
    result, _ = both(reactions_func, (100.0, 200.0), init_state.copy(), method=method,
                     dense_output=True, max_step=100.0)
    assert result.success


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("t0,t_bound", [(5, 9), (5, 1)])
def test_inital_maxstep(cls, t0, t_bound):
    # Adapted: scipy calls the private select_initial_step; we observe the
    # selected step through the solver's h_abs.
    y0 = np.array([1 / 3, 2 / 9])
    step_no_max = cls(fun_rational, t0, y0, t_bound, rtol=1e-3, atol=1e-6).h_abs
    max_step = step_no_max / 2
    step_with_max = cls(fun_rational, t0, y0, t_bound, rtol=1e-3, atol=1e-6, max_step=max_step).h_abs
    assert_equal(max_step, step_with_max)


def test_linear_oscillator_exact():
    # fun_linear/sol_linear are defined in scipy's test module; exercise them here.
    for method in METHODS:
        res, _ = both(fun_linear, [0, 10], sol_linear(0.0).ravel(), method=method, rtol=1e-8, atol=1e-10)
        assert_array_less(np.abs(res.y - sol_linear(res.t)), 1e-5)
