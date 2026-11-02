"""Event-location edge cases.

Every case runs through SciPy, our engine with Python events, and our engine
with native events (where expressible), and all three must agree. Some cases
document *limitations shared with SciPy*: sign-change detection cannot see a
root pair inside one step, or a grazing touch. Those tests pin the behaviour
down rather than call it a failure.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp as scipy_solve_ivp

from fast_numerical_integration.integrators import ComponentCrossing, solve_ivp
from fast_numerical_integration.problems.native import LinearSystem, TimeEvent

METHODS = ("RK23", "RK45", "DOP853")


def zero_rhs(t, y):
    return np.zeros_like(y)


def unit_drift(t, y):
    return np.ones_like(y)


def _event(fun, terminal=None, direction=0):
    fun.terminal = terminal
    fun.direction = direction
    return fun


def _time_event(t_e, terminal=None, direction=0):
    return _event(lambda t, y: t - t_e, terminal, direction)


def both(*args, **kwargs):
    ref = scipy_solve_ivp(*args, **kwargs)
    res = solve_ivp(*args, **kwargs)
    assert res.status == ref.status
    assert res.nfev == ref.nfev
    assert [t.size for t in res.t_events] == [t.size for t in ref.t_events]
    for a, b in zip(res.t_events, ref.t_events):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)
    return res, ref


# --- boundaries of the interval ------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_event_exactly_at_t0(method):
    # g(t0) = 0 and g increasing: scipy's rule (g <= 0 <= g_new) reports t0.
    res, _ = both(unit_drift, (0.0, 2.0), [0.0], method=method,
                  events=_event(lambda t, y: y[0]))
    np.testing.assert_array_equal(res.t_events[0], [0.0])
    # With direction = -1 the same zero at t0 is not a crossing.
    res, _ = both(unit_drift, (0.0, 2.0), [0.0], method=method,
                  events=_event(lambda t, y: y[0], direction=-1))
    assert res.t_events[0].size == 0


@pytest.mark.parametrize("method", METHODS)
def test_event_exactly_at_t_end(method):
    res, _ = both(zero_rhs, (0.0, 2.0), [1.0], method=method, events=_time_event(2.0))
    np.testing.assert_array_equal(res.t_events[0], [2.0])
    native = solve_ivp(zero_rhs, (0.0, 2.0), [1.0], method=method, events=TimeEvent(2.0))
    np.testing.assert_array_equal(native.t_events[0], [2.0])


@pytest.mark.parametrize("method", METHODS)
def test_terminal_event_exactly_at_t_end(method):
    res, _ = both(zero_rhs, (0.0, 2.0), [1.0], method=method, events=_time_event(2.0, terminal=True))
    assert res.status == 1
    assert res.t[-1] == 2.0


# --- several events in a single step ----------------------------------------------------------

def _single_big_step_kw():
    # Zero RHS: the error estimate is 0, so steps grow 10x each time; the step
    # crossing t ~ 0.1 .. 1.1 contains all three event times below.
    return dict(t_span=(0.0, 10.0), y0=[1.0])


@pytest.mark.parametrize("method", METHODS)
def test_simultaneous_events_ordered_and_truncated_at_terminal(method):
    """Events at 0.5 (plain), 0.6 (terminal), 0.7 (plain), all in one step:
    the 0.7 root comes after termination and must be dropped."""
    kw = _single_big_step_kw()
    evs = [_time_event(0.7), _time_event(0.6, terminal=True), _time_event(0.5)]
    res, _ = both(zero_rhs, kw["t_span"], kw["y0"], method=method, events=evs)
    assert res.status == 1
    assert res.t_events[0].size == 0
    np.testing.assert_allclose(res.t_events[1], [0.6])
    np.testing.assert_allclose(res.t_events[2], [0.5])
    assert res.t[-1] == pytest.approx(0.6)
    native = solve_ivp(zero_rhs, kw["t_span"], kw["y0"], method=method,
                       events=[TimeEvent(0.7), TimeEvent(0.6, terminal=True), TimeEvent(0.5)])
    for a, b in zip(native.t_events, res.t_events):
        np.testing.assert_allclose(a, b, rtol=1e-15)


@pytest.mark.parametrize("method", METHODS)
def test_simultaneous_events_backward(method):
    evs = [_time_event(-0.7), _time_event(-0.6, terminal=True), _time_event(-0.5)]
    res, _ = both(zero_rhs, (0.0, -10.0), [1.0], method=method, events=evs)
    assert res.status == 1
    assert res.t_events[0].size == 0
    np.testing.assert_allclose(res.t_events[1], [-0.6])
    np.testing.assert_allclose(res.t_events[2], [-0.5])


@pytest.mark.parametrize("method", METHODS)
def test_two_terminal_events_earliest_wins(method):
    evs = [_time_event(0.8, terminal=True), _time_event(0.4, terminal=True)]
    res, _ = both(zero_rhs, (0.0, 10.0), [1.0], method=method, events=evs)
    assert res.t[-1] == pytest.approx(0.4)
    assert res.t_events[0].size == 0


# --- limitations shared with scipy ---------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_root_pair_inside_one_step_is_missed(method):
    """g = (t - 0.5)(t - 0.6): both roots fall in one step, g has the same sign
    at both ends, so sign-change detection sees nothing. Same in scipy."""
    res, _ = both(zero_rhs, (0.0, 10.0), [1.0], method=method,
                  events=_event(lambda t, y: (t - 0.5) * (t - 0.6)))
    assert res.t_events[0].size == 0


@pytest.mark.parametrize("method", METHODS)
def test_root_pair_found_when_steps_are_limited(method):
    """The fix for the above: cap max_step below the root spacing."""
    res, _ = both(zero_rhs, (0.0, 1.0), [1.0], method=method, max_step=0.05,
                  events=_event(lambda t, y: (t - 0.5) * (t - 0.6)))
    np.testing.assert_allclose(res.t_events[0], [0.5, 0.6])


@pytest.mark.parametrize("method", METHODS)
def test_odd_number_of_roots_in_one_step_reports_one(method):
    g = _event(lambda t, y: (t - 0.4) * (t - 0.5) * (t - 0.6))
    res, _ = both(zero_rhs, (0.0, 10.0), [1.0], method=method, events=g)
    assert res.t_events[0].size == 1
    assert np.min(np.abs(res.t_events[0][0] - np.array([0.4, 0.5, 0.6]))) < 1e-12


@pytest.mark.parametrize("method", METHODS)
def test_grazing_touch_is_missed(method):
    """g = -(t - 1)^2 touches zero at t = 1 without changing sign."""
    res, _ = both(unit_drift, (0.0, 2.0), [0.0], method=method, max_step=0.1,
                  events=_event(lambda t, y: -(t - 1.0) ** 2 - 1e-12))
    assert res.t_events[0].size == 0


# --- unusual event functions ------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_discontinuous_event_function(method):
    """A step function: Brent's method converges onto the jump (y = 0.5 at t = 0.5)."""
    g = _event(lambda t, y: 1.0 if y[0] > 0.5 else -1.0)
    res, _ = both(unit_drift, (0.0, 2.0), [0.0], method=method, events=g)
    assert res.t_events[0][0] == pytest.approx(0.5, abs=1e-14)


@pytest.mark.parametrize("method", METHODS)
def test_time_event_located_to_machine_precision(method):
    for t_e in (0.1, 1.0 / 3.0, 7.77):
        res = solve_ivp(zero_rhs, (0.0, 10.0), [1.0], method=method, events=TimeEvent(t_e))
        assert res.t_events[0][0] == pytest.approx(t_e, rel=8 * np.finfo(float).eps)


# --- many events, direction filters -------------------------------------------------------------

OSC = np.array([[0.0, 1.0], [-1.0, 0.0]])


def _oscillator_crossings(method, direction, n_periods, native, **kw):
    if native:
        fun, ev = LinearSystem(OSC), ComponentCrossing(0, direction=direction)
    else:
        fun, ev = (lambda t, y: OSC @ y), _event(lambda t, y: y[0], direction=direction)
    # x = sin t: zeros at k pi (t = 0 included, since x' > 0 there).
    return solve_ivp(fun, (0.0, 2 * np.pi * n_periods + 1.0), [0.0, 1.0], method=method,
                     events=ev, **kw)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("direction,expected_k", [(0, "all"), (1, "even"), (-1, "odd")])
@pytest.mark.parametrize("native", [True, False], ids=["native", "python"])
def test_direction_filters_on_long_oscillation(method, direction, expected_k, native):
    n = 50
    res = _oscillator_crossings(method, direction, n, native, rtol=1e-10, atol=1e-12)
    k = np.arange(2 * n + 1)
    if expected_k == "even":
        k = k[k % 2 == 0]
    elif expected_k == "odd":
        k = k[k % 2 == 1]
    np.testing.assert_allclose(res.t_events[0], k * np.pi, atol=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_native_and_python_events_agree_exactly(method):
    a = _oscillator_crossings(method, 0, 20, native=True, rtol=1e-8, atol=1e-10)
    b = _oscillator_crossings(method, 0, 20, native=False, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(a.t_events[0], b.t_events[0], rtol=0, atol=1e-12)
    assert a.nfev == b.nfev


@pytest.mark.parametrize("method", METHODS)
def test_backward_direction_semantics_match_scipy(method):
    """``direction`` refers to the sign change in the order integration visits
    g, so +1 integrating backwards finds the crossings where g decreases
    with t. Same as scipy."""
    for direction in (-1, 0, 1):
        both(lambda t, y: OSC @ y, (4 * np.pi + 0.5, 0.0), [np.sin(4 * np.pi + 0.5), np.cos(4 * np.pi + 0.5)],
             method=method, rtol=1e-9, atol=1e-12, events=_event(lambda t, y: y[0], direction=direction))


@pytest.mark.parametrize("method", METHODS)
def test_terminal_count_larger_than_available_crossings(method):
    res, _ = both(lambda t, y: OSC @ y, (0.0, 3 * np.pi + 0.5), [0.0, 1.0], method=method,
                  events=_event(lambda t, y: y[0], terminal=10))
    assert res.status == 0
    assert res.t_events[0].size == 4


@pytest.mark.parametrize("method", METHODS)
def test_event_with_t_eval_and_dense_output_after_termination(method):
    """After a terminal event: t_eval stops at the event; dense output covers
    up to the event; y at the event matches the interpolant."""
    t_eval = np.linspace(0.0, 10.0, 101)
    res, ref = both(lambda t, y: OSC @ y, (0.0, 10.0), [0.0, 1.0], method=method, t_eval=t_eval,
                    dense_output=True, rtol=1e-9, atol=1e-12,
                    events=_event(lambda t, y: y[0], terminal=3))
    t_stop = res.t_events[0][-1]
    assert t_stop == pytest.approx(2 * np.pi, abs=1e-7)
    assert res.t.max() <= t_stop
    np.testing.assert_allclose(res.sol(res.t_events[0]), res.y_events[0].T, atol=1e-12)
    np.testing.assert_array_equal(res.t, ref.t)
