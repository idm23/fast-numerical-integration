"""Event detection beyond scipy: splitting a step at located roots, and
``event_detection="interior"``.

Most cases integrate ``y' = 1`` over (0, 10) in a single step
(``first_step=10``: the error estimate is zero, so the step is accepted), so
exactly what the detector sees inside one step is under test. ``y = t``, so an
event written in ``y`` has its roots at known times.
"""

import numpy as np
import pytest

from fast_numerical_integration.integrators import (ComponentCrossing, ScaleComponent, integrate,
                                                    integrate_hybrid, solve_hybrid, solve_ivp)
from fast_numerical_integration.problems import native

METHODS = ("RK23", "RK45", "DOP853")
ONE_STEP = dict(first_step=10.0)


def unit_drift(t, y):
    return np.ones_like(y)


def _event(fun, terminal=None, direction=0):
    fun.terminal = terminal
    fun.direction = direction
    return fun


def _pair(lo, hi, **kw):
    """Negative on (lo, hi), positive outside: a pair of crossings."""
    return _event(lambda t, y: (y[0] - lo) * (y[0] - hi), **kw)


def _at(t_e, **kw):
    return _event(lambda t, y: y[0] - t_e, **kw)


def _keep(e, t, y):
    return y


# --- splitting a step at located roots ------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_scipy_misses_both_roots_of_a_pair_in_one_step(method):
    """The baseline: a pair inside one step, with no other event, is unseen."""
    res = solve_ivp(unit_drift, (0, 10), [0.0], method=method, events=_pair(4, 8), **ONE_STEP)
    assert res.nfev < 30 and res.t_events[0].size == 0


@pytest.mark.parametrize("method", METHODS)
def test_a_root_splits_the_step_for_other_events(method):
    """A non-terminal event at 6 splits the step: the pair (4, 8) around it,
    invisible at the step ends, is found on both pieces, in time order."""
    res = solve_ivp(unit_drift, (0, 10), [0.0], method=method,
                    events=[_at(6), _pair(4, 8)], **ONE_STEP)
    np.testing.assert_allclose(res.t_events[0], [6.0], rtol=1e-14)
    np.testing.assert_allclose(res.t_events[1], [4.0, 8.0], rtol=1e-14)
    out = integrate(unit_drift, (0, 10), [0.0], method=method,
                    events=[_at(6), _pair(4, 8)], **ONE_STEP)
    np.testing.assert_array_equal(out.t_events[1], res.t_events[1])


@pytest.mark.parametrize("method", METHODS)
def test_hidden_terminal_root_before_another_root_ends_the_step(method):
    """The terminal event at 4 is hidden at the step ends but comes before the
    root at 6: the integration stops at 4, and the event at 6 never happens."""
    res = solve_ivp(unit_drift, (0, 10), [0.0], method=method,
                    events=[_at(6), _pair(4, 8, terminal=True)], **ONE_STEP)
    assert res.status == 1
    assert res.t[-1] == pytest.approx(4.0, rel=1e-14)
    assert res.t_events[0].size == 0
    np.testing.assert_allclose(res.t_events[1], [4.0], rtol=1e-14)


@pytest.mark.parametrize("method", METHODS)
def test_crossing_before_a_reset_is_not_lost(method):
    """A guard at 6 cuts the step. The other event crossed at 4, inside the
    part that was integrated, and crosses back at 8, after the reset. Both
    crossings are found; before, the one at 4 was dropped at the reset."""
    for driver in (solve_hybrid, integrate_hybrid):
        res = driver(unit_drift, (0, 10), [0.0], [_at(6, terminal=True), _pair(4, 8)], _keep,
                     method=method, **ONE_STEP)
        assert res.status == 0
        np.testing.assert_allclose(res.t_resets, [6.0], rtol=1e-14)
        np.testing.assert_allclose(res.t_events[1], [4.0, 8.0], rtol=1e-14)


@pytest.mark.parametrize("method", METHODS)
def test_direction_applies_to_split_pieces(method):
    res = solve_ivp(unit_drift, (0, 10), [0.0], method=method,
                    events=[_at(6), _pair(4, 8, direction=1)], **ONE_STEP)
    np.testing.assert_allclose(res.t_events[1], [8.0], rtol=1e-14)


@pytest.mark.parametrize("method", METHODS)
def test_simultaneous_roots_are_reported_in_index_order(method):
    res = integrate(unit_drift, (0, 10), [0.0], method=method,
                    events=[_at(5), _at(5), _pair(3, 7)], **ONE_STEP)
    np.testing.assert_allclose(res.t_events[0], [5.0])
    np.testing.assert_allclose(res.t_events[1], [5.0])
    np.testing.assert_allclose(res.t_events[2], [3.0, 7.0], rtol=1e-14)


# --- event_detection="interior" -------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_interior_points_find_a_pair_in_one_step(method):
    for driver in (solve_ivp, integrate):
        res = driver(unit_drift, (0, 10), [0.0], method=method, events=_pair(4, 8),
                     event_detection="interior", n_interior=1, **ONE_STEP)
        np.testing.assert_allclose(res.t_events[0], [4.0, 8.0], rtol=1e-14)


@pytest.mark.parametrize("method", METHODS)
def test_interior_points_resolve_pairs_down_to_their_spacing(method):
    """Test points at k * 10 / (n + 1): the pair (4.4, 4.6) is found once a
    point lands inside it, and missed while none does."""
    kw = dict(method=method, events=_pair(4.4, 4.6), event_detection="interior", **ONE_STEP)
    assert solve_ivp(unit_drift, (0, 10), [0.0], n_interior=4, **kw).t_events[0].size == 0
    res = solve_ivp(unit_drift, (0, 10), [0.0], n_interior=19, **kw)   # a point at 4.5
    np.testing.assert_allclose(res.t_events[0], [4.4, 4.6], rtol=1e-14)


@pytest.mark.parametrize("method", METHODS)
def test_interior_backward_in_time(method):
    res = solve_ivp(unit_drift, (10, 0), [10.0], method=method, events=_pair(4, 8),
                    event_detection="interior", first_step=10.0)
    np.testing.assert_allclose(res.t_events[0], [8.0, 4.0], rtol=1e-14)


@pytest.mark.parametrize("method", METHODS)
def test_interior_terminal_pair_stops_at_the_first_root(method):
    res = solve_ivp(unit_drift, (0, 10), [0.0], method=method,
                    events=_pair(4, 8, terminal=True), event_detection="interior", **ONE_STEP)
    assert res.status == 1 and res.t[-1] == pytest.approx(4.0, rel=1e-14)


def test_interior_cost():
    """DOP853 builds its interpolant with 3 extra RHS evaluations per step;
    RK23 and RK45 build theirs from the stages they already have."""
    def osc(t, y):
        return [y[1], -y[0]]

    ev = _event(lambda t, y: y[0] - 2.0)       # never crosses: no interpolant needed
    for method, extra in (("RK23", 0), ("RK45", 0), ("DOP853", 3)):
        a = solve_ivp(osc, (0, 20), [1.0, 0.0], method=method, events=ev, rtol=1e-8)
        b = solve_ivp(osc, (0, 20), [1.0, 0.0], method=method, events=ev, rtol=1e-8,
                      event_detection="interior")
        assert b.n_accepted == a.n_accepted
        assert b.nfev == a.nfev + extra * a.n_accepted
        np.testing.assert_array_equal(b.y, a.y)       # same steps


def test_interior_agrees_with_endpoints_on_the_bouncing_ball():
    """Guards with reset maps (and their post-reset probes) under interior detection."""
    ball = native.LinearSystem([[0.0, 1.0], [0.0, 0.0]], [0.0, -9.81])

    def run(**kw):
        return solve_hybrid(ball, (0, 3.0), [1.0, 0.0],
                            ComponentCrossing(0, 0.0, terminal=True, direction=-1),
                            ScaleComponent(1, -0.8), method="DOP853", rtol=1e-10, atol=1e-12,
                            **kw)

    a, b = run(), run(event_detection="interior", n_interior=8)
    assert a.t_resets.size == b.t_resets.size > 5
    np.testing.assert_allclose(b.t_resets, a.t_resets, rtol=1e-12)


def test_validation():
    with pytest.raises(ValueError, match="event_detection"):
        solve_ivp(unit_drift, (0, 1), [0.0], event_detection="everywhere")
    with pytest.raises(ValueError, match="n_interior"):
        solve_ivp(unit_drift, (0, 1), [0.0], event_detection="interior", n_interior=0)
    with pytest.raises(ValueError, match="event_detection"):
        integrate(unit_drift, (0, 1), [0.0], event_detection=None)


@pytest.mark.parametrize("method", METHODS)
def test_roots_on_test_points_are_reported_once(method):
    """n_interior=4 puts test points at 2, 4, 6, 8: the roots 4 and 8 sit
    exactly on them, where g is exactly zero."""
    for span, y0, expected in (((0, 10), 0.0, [4.0, 8.0]), ((10, 0), 10.0, [8.0, 4.0])):
        res = solve_ivp(unit_drift, span, [y0], method=method, events=_pair(4, 8),
                        event_detection="interior", n_interior=4, first_step=10.0)
        np.testing.assert_allclose(res.t_events[0], expected, rtol=1e-14)


@pytest.mark.parametrize("method", METHODS)
def test_interior_roots_are_true_roots_and_extend_endpoint_roots(method):
    """y = cos t with loose tolerances (long steps): events y = c for c near 1
    cross in pairs close to each peak. Interior detection reports only roots
    of the computed trajectory, in order, and every root endpoint detection
    reports (both take the same steps). RK23 at this tolerance damps the
    amplitude, so its peaks never return close to 1 and it misses no pairs;
    the one-step tests above cover it."""
    def osc(t, y):
        return [y[1], -y[0]]

    levels = (0.5, 0.9, 0.97, 0.995)
    evs = [_event(lambda t, y, c=c: y[0] - c) for c in levels]
    kw = dict(method=method, events=evs, rtol=1e-3, atol=1e-6, dense_output=True)
    a = solve_ivp(osc, (0, 40), [1.0, 0.0], **kw)
    b = solve_ivp(osc, (0, 40), [1.0, 0.0], event_detection="interior", n_interior=6, **kw)
    found_more = False
    for c, ta, tb in zip(levels, a.t_events, b.t_events):
        assert np.all(np.diff(tb) > 0)
        np.testing.assert_allclose(b.sol(tb)[0], c, atol=1e-12)
        assert all(np.min(np.abs(tb - x)) < 1e-12 for x in ta)
        found_more |= tb.size > ta.size
    assert found_more or method == "RK23"
