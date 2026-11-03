"""Hybrid integration: guards, reset maps, modes, Zeno limits.

``solve_hybrid`` (Python driver) and ``integrate_hybrid`` (``fni_integrate_hybrid``)
share ``fni_hybrid_reset``, so they must agree bit for bit, as ``solve_ivp`` and
``integrate`` do. Accuracy is checked against closed forms: a bouncing ball
(polynomial between impacts, so located to round-off) and a thermostat with
hysteresis (exponential arcs).
"""

import threading

import numpy as np
import pytest

from fast_numerical_integration.integrators import (ComponentCrossing, ResetMap, ScaleComponent,
                                                    integrate, integrate_hybrid, solve_hybrid,
                                                    solve_ivp)
from fast_numerical_integration.problems import native

G, H0, E = 9.81, 1.0, 0.8
METHODS = ("RK23", "RK45", "DOP853")


def ball():
    """y = [height, velocity]: free fall as a native affine system."""
    return native.LinearSystem([[0.0, 1.0], [0.0, 0.0]], [0.0, -G])


def ground(direction=-1.0):
    return ComponentCrossing(0, 0.0, terminal=True, direction=direction)


def bounce_times(k, e=E, h0=H0):
    """First k impact times from rest at h0 with restitution e."""
    t1 = np.sqrt(2 * h0 / G)
    v1 = G * t1
    gaps = [2 * v1 * e ** i / G for i in range(1, k)]
    return t1 + np.concatenate([[0.0], np.cumsum(gaps)])


def zeno_time(e=E, h0=H0):
    t1 = np.sqrt(2 * h0 / G)
    return t1 + 2 * G * t1 * e / (G * (1 - e))


@pytest.mark.parametrize("method", METHODS)
def test_bouncing_ball_matches_closed_form(method):
    res = solve_hybrid(ball(), (0, 3.0), [H0, 0.0], ground(), ScaleComponent(1, -E),
                       method=method, rtol=1e-10, atol=1e-12)
    assert res.status == 0
    k = len(res.t_resets)
    expected = bounce_times(k)
    assert expected[-1] < 3.0 < bounce_times(k + 1)[-1]
    np.testing.assert_allclose(res.t_resets, expected, rtol=1e-9)
    np.testing.assert_array_equal(res.reset_events, 0)
    np.testing.assert_allclose(res.y_after_resets[:, 1], -E * res.y_before_resets[:, 1])
    assert np.all(res.y_before_resets[:, 1] < 0) and np.all(res.y_after_resets[:, 1] > 0)
    # The trajectory holds each impact twice: before, then after.
    for tr in res.t_resets:
        assert np.count_nonzero(res.t == tr) == 2
    # The height never goes (meaningfully) below the ground.
    assert res.y[0].min() > -1e-9


@pytest.mark.parametrize("method", METHODS)
def test_endpoint_driver_is_bit_identical(method):
    kw = dict(method=method, rtol=1e-8, atol=1e-10)
    a = solve_hybrid(ball(), (0, 2.5), [H0, 0.0], ground(), ScaleComponent(1, -E), **kw)
    b = integrate_hybrid(ball(), (0, 2.5), [H0, 0.0], ground(), ScaleComponent(1, -E), **kw)
    assert b.gil_released
    assert a.nfev == b.nfev and a.n_accepted == b.n_accepted
    np.testing.assert_array_equal(a.y[:, -1], b.y)
    np.testing.assert_array_equal(a.t_resets, b.t_resets)
    np.testing.assert_array_equal(a.y_before_resets, b.y_before_resets)
    np.testing.assert_array_equal(a.y_after_resets, b.y_after_resets)
    assert b.n_resets == len(b.t_resets)


def test_python_reset_matches_native():
    kw = dict(method="DOP853", rtol=1e-9, atol=1e-12)
    a = solve_hybrid(ball(), (0, 2.5), [H0, 0.0], ground(), ScaleComponent(1, -E), **kw)
    b = solve_hybrid(ball(), (0, 2.5), [H0, 0.0], ground(),
                     lambda k, t, y: [y[0], -E * y[1]], **kw)
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.t_resets, b.t_resets)
    c = integrate_hybrid(ball(), (0, 2.5), [H0, 0.0], ground(),
                         lambda k, t, y: [y[0], -E * y[1]], **kw)
    assert not c.gil_released
    np.testing.assert_array_equal(a.y[:, -1], c.y)


def test_step_size_is_carried_across_resets():
    kw = dict(method="DOP853", rtol=1e-10, atol=1e-12)
    keep = integrate_hybrid(ball(), (0, 2.5), [H0, 0.0], ground(), ScaleComponent(1, -E), **kw)
    fresh = integrate_hybrid(ball(), (0, 2.5), [H0, 0.0], ground(), ScaleComponent(1, -E),
                             reselect_step=True, **kw)
    np.testing.assert_allclose(keep.t_resets, fresh.t_resets, rtol=1e-10)
    np.testing.assert_allclose(keep.y, fresh.y, rtol=1e-8, atol=1e-10)
    # Restarting the step-size heuristic costs evaluations at every reset.
    assert keep.nfev < fresh.nfev


def test_guard_is_masked_after_its_reset():
    """A both-directions guard sits on zero right after the bounce; it must not
    fire again there, so the result equals the one-direction guard's."""
    kw = dict(method="RK45", rtol=1e-9, atol=1e-12)
    down = solve_hybrid(ball(), (0, 2.5), [H0, 0.0], ground(-1.0), ScaleComponent(1, -E), **kw)
    both = solve_hybrid(ball(), (0, 2.5), [H0, 0.0], ground(0.0), ScaleComponent(1, -E), **kw)
    np.testing.assert_array_equal(down.t_resets, both.t_resets)
    np.testing.assert_array_equal(down.y, both.y)


def test_zeno_and_reset_limit():
    tz = zeno_time(e=0.5)
    res = integrate_hybrid(ball(), (0, 10.0), [H0, 0.0], ground(), ScaleComponent(1, -0.5),
                           method="DOP853", rtol=1e-10, atol=1e-12, min_dwell=1e-6,
                           max_resets=None)
    assert res.status == -3 and not res.success
    assert "Zeno" in res.message
    assert abs(res.t - tz) < 1e-5
    np.testing.assert_allclose(res.t_resets, bounce_times(len(res.t_resets), e=0.5), rtol=1e-8)

    res = integrate_hybrid(ball(), (0, 10.0), [H0, 0.0], ground(), ScaleComponent(1, -0.5),
                           method="DOP853", rtol=1e-10, atol=1e-12, max_resets=5)
    assert res.status == -2 and res.n_resets == 5
    # Stopped at the 6th impact, before applying it.
    assert abs(res.t - bounce_times(6, e=0.5)[-1]) < 1e-9
    assert res.y[1] < 0

    res = integrate_hybrid(ball(), (0, 10.0), [H0, 0.0], ground(), ScaleComponent(1, -0.5),
                           max_resets=0)
    assert res.status == -2 and res.n_resets == 0


def test_reset_can_stop():
    def stop_on_third(k, t, y, counter=[0]):
        counter[0] += 1
        return None if counter[0] == 3 else [y[0], -E * y[1]]

    res = solve_hybrid(ball(), (0, 10.0), [H0, 0.0], ground(), stop_on_third,
                       method="DOP853", rtol=1e-10, atol=1e-12)
    assert res.status == 1 and res.success
    assert abs(res.t[-1] - bounce_times(3)[-1]) < 1e-9
    assert len(res.t_resets) == 3     # the reset map was called three times


class Thermostat:
    """x' = -x + 2 * heating; on above 1.5 -> off, off below 0.5 -> on."""

    def __init__(self):
        self.heating = 1

    def rhs(self, t, y):
        return [-y[0] + 2.0 * self.heating]

    def guard(self, t, y):
        return y[0] - (1.5 if self.heating else 0.5)

    def reset(self, k, t, y):
        self.heating = 1 - self.heating
        return y


@pytest.mark.parametrize("method", METHODS)
def test_modes_with_a_mode_dependent_guard(method):
    th = Thermostat()
    guard = th.guard
    guard.__func__.terminal = True
    res = solve_hybrid(th.rhs, (0, 10.0), [0.5], guard, th.reset, method=method,
                       rtol=1e-11, atol=1e-12, dense_output=True)
    # Each half-cycle (heating 0.5 -> 1.5, cooling 1.5 -> 0.5) takes ln 3.
    k = len(res.t_resets)
    np.testing.assert_allclose(res.t_resets, np.log(3) * np.arange(1, k + 1), rtol=1e-8)
    assert k == int(10.0 / np.log(3))
    # The guard jumps from 0 to +-1 at each reset; that jump is not a crossing.
    np.testing.assert_allclose(res.y_before_resets[:, 0], [1.5, 0.5] * (k // 2) + [1.5] * (k % 2),
                               rtol=1e-9)
    # Dense output is piecewise across resets.
    ts = np.linspace(0, 10, 101)
    phase = np.mod(ts, 2 * np.log(3))
    exact = np.where(phase < np.log(3), 2 - 1.5 * np.exp(-phase),
                     1.5 * np.exp(-(phase - np.log(3))))
    np.testing.assert_allclose(res.sol(ts)[0], exact, rtol=1e-7)


def test_t_eval_and_non_terminal_events():
    t_eval = np.linspace(0, 2.5, 26)
    apex = ComponentCrossing(1, 0.0, direction=-1)       # velocity + -> -: recorded only
    res = solve_hybrid(ball(), (0, 2.5), [H0, 0.0], [ground(), apex], ScaleComponent(1, -E),
                       t_eval=t_eval, method="DOP853", rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(res.t, t_eval)
    np.testing.assert_array_equal(res.reset_events, 0)
    impacts = bounce_times(len(res.t_resets))
    # Apexes lie halfway between consecutive impacts. (The first record is
    # t = 0, where the velocity starts at zero, as in scipy.)
    assert res.t_events[1][0] == 0.0
    np.testing.assert_allclose(res.t_events[1][1:len(impacts)],
                               (impacts[:-1] + impacts[1:]) / 2, rtol=1e-9)

    def exact_height(t):
        t1 = impacts[0]
        if t <= t1:
            return H0 - G * t ** 2 / 2
        i = np.searchsorted(impacts, t) - 1
        v = G * t1 * E ** (i + 1)
        s = t - impacts[i]
        return v * s - G * s ** 2 / 2

    np.testing.assert_allclose(res.y[0], [exact_height(t) for t in t_eval], atol=1e-9)


def test_dense_output_across_resets():
    res = solve_hybrid(ball(), (0, 2.5), [H0, 0.0], ground(), ScaleComponent(1, -E),
                       method="DOP853", rtol=1e-11, atol=1e-12, dense_output=True)
    sol = res.sol
    for tr, before, after in zip(res.t_resets, res.y_before_resets, res.y_after_resets):
        np.testing.assert_array_equal(sol(tr), before)        # at a reset: the state before
        assert sol(tr + 1e-9)[1] > 0                           # just after: moving up again
        np.testing.assert_allclose(sol(tr + 1e-9), after, atol=1e-7)


def test_without_resets_matches_solve_ivp():
    """No terminal events: the hybrid drivers reduce to the plain ones."""
    apex = ComponentCrossing(1, 0.0)
    kw = dict(method="DOP853", rtol=1e-8, atol=1e-10)
    a = solve_ivp(ball(), (0, 0.4), [H0, 0.0], events=apex, **kw)
    b = solve_hybrid(ball(), (0, 0.4), [H0, 0.0], apex, ScaleComponent(1, -E), **kw)
    c = integrate(ball(), (0, 0.4), [H0, 0.0], events=apex, **kw)
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.y[:, -1], c.y)
    assert a.nfev == b.nfev == c.nfev
    assert len(b.t_resets) == 0


def test_no_reset_map_stops_like_solve_ivp():
    kw = dict(method="RK45", rtol=1e-8, atol=1e-10)
    a = solve_ivp(ball(), (0, 2.0), [H0, 0.0], events=ground(), **kw)
    b = solve_hybrid(ball(), (0, 2.0), [H0, 0.0], ground(), None, **kw)
    assert a.status == b.status == 1
    np.testing.assert_array_equal(a.y, b.y)


def test_backward_in_time():
    """Integrating back from a point after the second impact recovers it."""
    fwd = solve_hybrid(ball(), (0, 1.5), [H0, 0.0], ground(), ScaleComponent(1, -E),
                       method="DOP853", rtol=1e-12, atol=1e-13)
    assert len(fwd.t_resets) == 2
    back = solve_hybrid(ball(), (1.5, 0.0), fwd.y[:, -1], ground(-1.0),
                        ScaleComponent(1, -1 / E), method="DOP853", rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(back.t_resets, fwd.t_resets[::-1], rtol=1e-9)
    np.testing.assert_allclose(back.y[:, -1], [H0, 0.0], atol=1e-8)


def test_runs_in_parallel_threads():
    out = {}

    def work(i):
        out[i] = integrate_hybrid(ball(), (0, 2.5), [H0 + 0.1 * i, 0.0], ground(),
                                  ScaleComponent(1, -E), method="DOP853", rtol=1e-10,
                                  atol=1e-12)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for i, res in out.items():
        assert res.gil_released
        np.testing.assert_allclose(res.t_resets, bounce_times(len(res.t_resets), h0=H0 + 0.1 * i),
                                   rtol=1e-9)


def test_validation():
    with pytest.raises(IndexError):
        solve_hybrid(ball(), (0, 1.0), [H0, 0.0], ground(), ScaleComponent(5, -E))
    with pytest.raises(ValueError, match="min_dwell"):
        solve_hybrid(ball(), (0, 1.0), [H0, 0.0], ground(), ScaleComponent(1, -E), min_dwell=-1)
    with pytest.raises(ValueError, match="reset returned"):
        solve_hybrid(ball(), (0, 1.0), [H0, 0.0], ground(), lambda k, t, y: [0.0])
    with pytest.raises(TypeError, match="release_gil"):
        integrate_hybrid(ball(), (0, 1.0), [H0, 0.0], ground(), lambda k, t, y: y,
                         release_gil=True)
    with pytest.raises(NotImplementedError):
        ResetMap()(0, 0.0, [1.0])
    r = ScaleComponent(1, -E)
    np.testing.assert_allclose(r(0, 0.0, [0.0, -2.0]), [0.0, 2.0 * E])
    r.params = [-0.5]
    np.testing.assert_allclose(r(0, 0.0, [0.0, -2.0]), [0.0, 1.0])
