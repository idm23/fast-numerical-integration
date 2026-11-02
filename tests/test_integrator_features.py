"""Tests for the solve_ivp features beyond plain stepping.

Covers t_eval, dense output (OdeSolution), events, vectorized, per-component
rtol, option validation and the stepper API, all checked against
:func:`scipy.integrate.solve_ivp`. The RHS evaluation counts must match
exactly. Interpolated values must agree to round-off, since both engines take
the same steps and use the same interpolants.
"""

import numpy as np
import pytest
import scipy.integrate as si
from scipy.integrate import solve_ivp as scipy_solve_ivp

from fast_numerical_integration import integrators
from fast_numerical_integration.integrators import (
    ComponentCrossing,
    EventFunction,
    OdeSolution,
    solve_ivp,
)
from fast_numerical_integration.three_body import figure_eight_system

METHODS = ("RK23", "RK45", "DOP853")
TOL = dict(rtol=1e-9, atol=1e-11)


@pytest.fixture(scope="module")
def system():
    return figure_eight_system()


def _spans(system):
    return [(0.0, system.period), (system.period, 0.0)]


def _close(a, b, atol=1e-10):
    np.testing.assert_allclose(a, b, rtol=0, atol=atol)


# --- t_eval -------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
def test_t_eval_matches_scipy(system, method, backward):
    span = _spans(system)[backward]
    t_eval = np.linspace(span[0], span[1], 57)
    ref = scipy_solve_ivp(system.rhs, span, system.y0, method=method, t_eval=t_eval, **TOL)
    sol = solve_ivp(system.native_rhs(), span, system.y0, method=method, t_eval=t_eval, **TOL)
    assert sol.nfev == ref.nfev
    np.testing.assert_array_equal(sol.t, ref.t)
    assert sol.y.shape == ref.y.shape == (12, 57)
    _close(sol.y, ref.y)
    assert sol.sol is None and sol.t_events is None and sol.y_events is None


def test_t_eval_subset_of_interval(system):
    t_eval = [1.0, 2.0, 2.5]
    ref = scipy_solve_ivp(system.rhs, (0.0, system.period), system.y0, method="DOP853", t_eval=t_eval, **TOL)
    sol = solve_ivp(system.rhs, (0.0, system.period), system.y0, method="DOP853", t_eval=t_eval, **TOL)
    assert sol.nfev == ref.nfev
    _close(sol.y, ref.y)


def test_t_eval_validation():
    f = lambda t, y: -y
    with pytest.raises(ValueError, match="1-dimensional"):
        solve_ivp(f, (0, 1), [1.0], t_eval=[[0.5]])
    with pytest.raises(ValueError, match="within `t_span`"):
        solve_ivp(f, (0, 1), [1.0], t_eval=[0.5, 1.5])
    with pytest.raises(ValueError, match="properly sorted"):
        solve_ivp(f, (0, 1), [1.0], t_eval=[0.5, 0.2])
    with pytest.raises(ValueError, match="properly sorted"):
        solve_ivp(f, (1, 0), [1.0], t_eval=[0.2, 0.5])


# --- dense output ---------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
def test_dense_output_matches_scipy(system, method, backward):
    span = _spans(system)[backward]
    ref = scipy_solve_ivp(system.rhs, span, system.y0, method=method, dense_output=True, **TOL)
    sol = solve_ivp(system.native_rhs(), span, system.y0, method=method, dense_output=True, **TOL)
    assert sol.nfev == ref.nfev
    assert isinstance(sol.sol, OdeSolution)
    assert sol.sol.n_segments == ref.sol.n_segments
    np.testing.assert_allclose(sol.sol.ts, ref.sol.ts, rtol=1e-4, atol=1e-6)
    assert (sol.sol.t_min, sol.sol.t_max) == (ref.sol.t_min, ref.sol.t_max)

    # Random points, the breakpoints themselves, and slight extrapolation.
    rng = np.random.default_rng(1)
    tq = np.concatenate([rng.uniform(0, system.period, 300), ref.sol.ts[::7], [-0.01, system.period + 0.01]])
    _close(sol.sol(tq), ref.sol(tq), atol=1e-9)
    assert sol.sol(1.234).shape == (12,)
    _close(sol.sol(1.234), ref.sol(1.234), atol=1e-9)
    assert sol.sol(tq).shape == (12, tq.size)
    # The interpolant reproduces the stored step values.
    _close(sol.sol(sol.t), sol.y, atol=1e-12)


def test_dense_output_with_t_eval_uses_step_breakpoints(system):
    t_eval = np.linspace(0, system.period, 11)
    kw = dict(method="RK45", t_eval=t_eval, dense_output=True, **TOL)
    ref = scipy_solve_ivp(system.rhs, (0.0, system.period), system.y0, **kw)
    sol = solve_ivp(system.native_rhs(), (0.0, system.period), system.y0, **kw)
    assert sol.nfev == ref.nfev
    assert sol.t.shape == (11,)
    assert sol.sol.n_segments == ref.sol.n_segments
    _close(sol.sol(np.linspace(0, system.period, 99)), ref.sol(np.linspace(0, system.period, 99)), atol=1e-9)


def test_dense_output_zero_length_interval():
    ref = scipy_solve_ivp(lambda t, y: -y, (1.0, 1.0), [1.0, 2.0], dense_output=True)
    sol = solve_ivp(lambda t, y: -y, (1.0, 1.0), [1.0, 2.0], dense_output=True)
    np.testing.assert_array_equal(sol.t, ref.t)
    np.testing.assert_array_equal(sol.sol(1.0), ref.sol(1.0))
    np.testing.assert_array_equal(sol.sol([1.0, 1.0]), ref.sol([1.0, 1.0]))


def test_ode_solution_validation():
    with pytest.raises(ValueError, match="strictly"):
        OdeSolution([0.0, 1.0, 0.5], np.zeros((2, 2 + 1 + 4)), 0, 1, 4)
    with pytest.raises(ValueError, match="don't match"):
        OdeSolution([0.0, 1.0], np.zeros((2, 2 + 1 + 4)), 0, 1, 4)
    sol = solve_ivp(lambda t, y: -y, (0, 1), [1.0], dense_output=True)
    with pytest.raises(ValueError, match="1-D"):
        sol.sol(np.zeros((2, 2)))


# --- events --------------------------------------------------------------------

def _crossing_events():
    def y1(t, y):
        return y[1]

    def x2_up(t, y):
        return y[2]

    x2_up.direction = 1

    def vx1_down(t, y):
        return y[6]

    vx1_down.direction = -1
    return [y1, x2_up, vx1_down]


def _native_crossing_events():
    return [ComponentCrossing(1), ComponentCrossing(2, direction=1), ComponentCrossing(6, direction=-1)]


def _assert_events_match(sol, ref):
    assert len(sol.t_events) == len(ref.t_events)
    for te, te_ref, ye, ye_ref in zip(sol.t_events, ref.t_events, sol.y_events, ref.y_events):
        assert te.shape == te_ref.shape
        assert ye.shape == ye_ref.shape
        if te.size:
            _close(te, te_ref, atol=1e-11)
            _close(ye, ye_ref, atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
@pytest.mark.parametrize("native", [False, True], ids=["pyevent", "native"])
def test_events_match_scipy(system, method, backward, native):
    span = _spans(system)[backward]
    ref = scipy_solve_ivp(system.rhs, span, system.y0, method=method, events=_crossing_events(), **TOL)
    events = _native_crossing_events() if native else _crossing_events()
    sol = solve_ivp(system.native_rhs(), span, system.y0, method=method, events=events, **TOL)
    assert sol.status == ref.status == 0
    assert sol.nfev == ref.nfev
    assert sum(te.size for te in sol.t_events) > 0
    _assert_events_match(sol, ref)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("terminal", [True, 2, 3])
def test_terminal_events_match_scipy(system, method, terminal):
    def y1(t, y):
        return y[1]

    y1.terminal = terminal

    def x1_half(t, y):
        return y[0] - 0.5

    kw = dict(method=method, dense_output=True, t_eval=np.linspace(0, system.period, 200), **TOL)
    ref = scipy_solve_ivp(system.rhs, (0.0, system.period), system.y0, events=[y1, x1_half], **kw)
    native = [ComponentCrossing(1, terminal=terminal), ComponentCrossing(0, 0.5)]
    sol = solve_ivp(system.native_rhs(), (0.0, system.period), system.y0, events=native, **kw)

    assert sol.status == ref.status == 1
    assert sol.message == ref.message == "A termination event occurred."
    assert sol.success
    assert sol.nfev == ref.nfev
    assert sol.t_events[0].size == terminal
    _assert_events_match(sol, ref)
    np.testing.assert_array_equal(sol.t, ref.t)
    _close(sol.y, ref.y)
    assert sol.sol.n_segments == ref.sol.n_segments


def test_terminal_event_final_point_without_t_eval(system):
    def y1(t, y):
        return y[1]

    y1.terminal = True
    ref = scipy_solve_ivp(system.rhs, (0.0, system.period), system.y0, method="DOP853", events=y1, **TOL)
    sol = solve_ivp(system.native_rhs(), (0.0, system.period), system.y0, method="DOP853", events=y1, **TOL)
    assert sol.t[-1] == pytest.approx(ref.t[-1], abs=1e-12)
    assert sol.t[-1] == sol.t_events[0][0]
    _close(sol.y[:, -1], ref.y[:, -1], atol=1e-11)
    assert abs(sol.y[1, -1]) < 1e-12


def test_falling_ball_hits_ground_analytically():
    g, h0 = 9.81, 10.0

    def ground(t, y):
        return y[0]

    ground.terminal = True
    ground.direction = -1
    sol = solve_ivp(lambda t, y: [y[1], -g], (0, 10), [h0, 0.0], method="DOP853",
                    events=ground, rtol=1e-12, atol=1e-12)
    assert sol.status == 1
    assert sol.t_events[0][0] == pytest.approx(np.sqrt(2 * h0 / g), rel=1e-12)
    assert sol.y_events[0].shape == (1, 2)
    assert sol.y_events[0][0, 1] == pytest.approx(-np.sqrt(2 * g * h0), rel=1e-12)


def test_events_receive_args():
    seen = []

    def ev(t, y, k):
        seen.append(k)
        return y[0] - 0.5

    sol = solve_ivp(lambda t, y, k: -k * y, (0, 2), [1.0], events=ev, args=(1.3,))
    assert set(seen) == {1.3}
    assert sol.t_events[0][0] == pytest.approx(np.log(2) / 1.3, rel=1e-3)


def test_no_events_gives_empty_arrays_like_scipy():
    def never(t, y):
        return 1.0

    ref = scipy_solve_ivp(lambda t, y: -y, (0, 1), [1.0], events=never)
    sol = solve_ivp(lambda t, y: -y, (0, 1), [1.0], events=never)
    assert sol.t_events[0].shape == ref.t_events[0].shape == (0,)
    assert sol.y_events[0].shape == ref.y_events[0].shape


def test_invalid_terminal_rejected():
    def ev(t, y):
        return y[0]

    ev.terminal = 1.5
    with pytest.raises(ValueError, match="terminal"):
        solve_ivp(lambda t, y: -y, (0, 1), [1.0], events=ev)


def test_event_errors_propagate():
    def ev(t, y):
        raise KeyError("event failed")

    with pytest.raises(KeyError, match="event failed"):
        solve_ivp(lambda t, y: -y, (0, 1), [1.0], events=ev)


def test_base_event_is_abstract():
    with pytest.raises(NotImplementedError):
        solve_ivp(lambda t, y: -y, (0, 1), [1.0], events=EventFunction())


def test_component_crossing_callable_and_bounds():
    ev = ComponentCrossing(1, 2.0)
    assert ev(0.0, np.array([0.0, 5.0])) == 3.0
    with pytest.raises(IndexError):
        solve_ivp(lambda t, y: -y, (0, 1), [1.0], events=ComponentCrossing(4))


# --- other solve_ivp options ----------------------------------------------------

def test_vectorized_python_rhs(system):
    calls = []

    def f(t, y):
        calls.append(y.shape)
        return -0.5 * y

    ref = scipy_solve_ivp(f, (0, 2), [1.0, 2.0], vectorized=True)
    calls.clear()
    sol = solve_ivp(f, (0, 2), [1.0, 2.0], vectorized=True)
    assert set(calls) == {(2, 1)}
    assert sol.nfev == ref.nfev
    _close(sol.y, ref.y, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_per_component_rtol(system, method):
    rtol = np.geomspace(1e-10, 1e-6, 12)
    ref = scipy_solve_ivp(system.rhs, (0, system.period), system.y0, method=method, rtol=rtol, atol=1e-12)
    sol = solve_ivp(system.native_rhs(), (0, system.period), system.y0, method=method, rtol=rtol, atol=1e-12)
    assert sol.nfev == ref.nfev
    _close(sol.y[:, -1], ref.y[:, -1], atol=1e-8)


def test_default_method_is_rk45_like_scipy(system):
    ref = scipy_solve_ivp(system.rhs, (0, system.period), system.y0)
    sol = solve_ivp(system.native_rhs(), (0, system.period), system.y0)
    assert sol.nfev == ref.nfev


def test_method_may_be_a_solver_class(system):
    a = solve_ivp(system.native_rhs(), (0, 1), system.y0, method=integrators.DOP853)
    b = solve_ivp(system.native_rhs(), (0, 1), system.y0, method="DOP853")
    np.testing.assert_array_equal(a.y, b.y)


def test_warnings_like_scipy():
    with pytest.warns(UserWarning, match="rtol"):
        solve_ivp(lambda t, y: -y, (0, 1), [1.0], rtol=1e-20)
    with pytest.warns(UserWarning, match="`jac`"):
        solve_ivp(lambda t, y: -y, (0, 1), [1.0], jac=None)


def test_y0_validation():
    f = lambda t, y: -y
    with pytest.raises(ValueError, match="complex"):
        solve_ivp(f, (0, 1), [1.0 + 1j])
    with pytest.raises(ValueError, match="1-dimensional"):
        solve_ivp(f, (0, 1), [[1.0]])
    with pytest.raises(ValueError, match="finite"):
        solve_ivp(f, (0, 1), [np.nan])
    with pytest.raises(ValueError, match="rtol"):
        solve_ivp(f, (0, 1), [1.0, 2.0], rtol=[1e-3, 1e-3, 1e-3])
    with pytest.raises(TypeError, match="args"):
        solve_ivp(f, (0, 1), [1.0], args=1.0)


def test_result_supports_dict_and_attribute_access(system):
    sol = solve_ivp(system.native_rhs(), (0, 1), system.y0)
    assert sol["nfev"] == sol.nfev
    assert set(si.solve_ivp(system.rhs, (0, 1), system.y0).keys()) <= set(sol.keys())


# --- stepper API ----------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_stepper_dense_output_matches_scipy(system, method):
    ours = getattr(integrators, method)(system.native_rhs(), 0.0, system.y0, system.period, **TOL)
    theirs = getattr(si, method)(system.rhs, 0.0, system.y0, system.period, **TOL)
    assert ours.t_old is None and ours.step_size is None
    with pytest.raises(RuntimeError, match="Dense output"):
        ours.dense_output()
    for _ in range(5):
        ours.step()
        theirs.step()
    # Step sizes agree only to ~1e-6 relative: the embedded error estimate is
    # cancellation-heavy, so summation order (our loops vs. BLAS) shifts it.
    assert ours.t_old == pytest.approx(theirs.t_old, rel=1e-5)
    assert ours.step_size == pytest.approx(theirs.step_size, rel=1e-5)
    _close(ours.y, theirs.y, atol=1e-5)
    assert ours.nfev == theirs.nfev

    d_ours, d_theirs = ours.dense_output(), theirs.dense_output()
    # Matches scipy at the left end and our own step result at the right end.
    _close(d_ours(ours.t_old), d_theirs(theirs.t_old), atol=1e-5)
    _close(d_ours(ours.t), ours.y, atol=1e-12)
    # Same interpolant shape over the step, compared in step-relative coordinates.
    x = np.linspace(0, 1, 9)
    tq_ours = ours.t_old + x * (ours.t - ours.t_old)
    tq_theirs = theirs.t_old + x * (theirs.t - theirs.t_old)
    _close(d_ours(tq_ours), d_theirs(tq_theirs), atol=1e-5)
    if method == "DOP853":
        assert ours.nfev == theirs.nfev  # both spent 3 extra RHS calls on dense output
