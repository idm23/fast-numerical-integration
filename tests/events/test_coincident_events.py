"""Guards with a root at the same instant: each fires once, and integration goes on."""

import numpy as np
import pytest

from fast_numerical_integration.integrators import integrate_hybrid, solve_hybrid

METHODS = ("RK23", "RK45", "DOP853")
OPTIONS = dict(rtol=1e-10, atol=1e-12, max_resets=200)


def _guard(k, level=0.3):
    def guard(t, y):
        return y[k] - level
    guard.terminal = True
    return guard


def _keep(event, t, y):
    return y


@pytest.mark.parametrize("driver", [solve_hybrid, integrate_hybrid],
                         ids=["solve_hybrid", "integrate_hybrid"])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("flow", [lambda t: 1.0, np.cos], ids=["straight", "curved"])
@pytest.mark.parametrize("n", [2, 3, 6])
def test_each_guard_at_one_instant_fires_once(driver, method, flow, n):
    s = driver(lambda t, y: np.full(n, flow(t)), (0, 1), np.zeros(n),
               [_guard(k) for k in range(n)], _keep, method=method, **OPTIONS)
    assert s.status == 0
    assert sorted(s.reset_events) == list(range(n))
    assert np.ptp(s.t_resets) < 1e-9


@pytest.mark.parametrize("method", METHODS)
def test_coincident_guards_fire_again_on_later_crossings(method):
    # Up through the level near 0.3, and back down through it near 2.84.
    s = solve_hybrid(lambda t, y: np.full(2, np.cos(t)), (0, 3), np.zeros(2),
                     [_guard(0), _guard(1)], _keep, method=method, **OPTIONS)
    assert s.status == 0
    assert sorted(s.reset_events) == [0, 0, 1, 1]
    expected = np.array([np.arcsin(0.3)] * 2 + [np.pi - np.arcsin(0.3)] * 2)
    assert np.sort(s.t_resets) == pytest.approx(expected, abs=1e-7)


def test_a_reset_acts_on_the_state_each_guard_saw():
    # Each fired guard adds to its own component, so a lost guard would show in the result.
    def kick(event, t, y):
        y = y.copy()
        y[2 + event] += 1.0
        return y

    s = solve_hybrid(lambda t, y: np.array([1.0, 1.0, 0.0, 0.0]), (0, 1), np.zeros(4),
                     [_guard(0), _guard(1)], kick, **OPTIONS)
    assert s.status == 0
    assert s.y[2:, -1] == pytest.approx([1.0, 1.0])


def test_a_reset_that_clears_the_other_guard_does_not_fire_it():
    # The first guard's reset moves both states back below the level.
    def back(event, t, y):
        return y - 0.1

    s = solve_hybrid(lambda t, y: np.ones(2), (0, 0.32), np.zeros(2),
                     [_guard(0), _guard(1)], back, **OPTIONS)
    assert s.status == 0
    assert len(s.t_resets) == 1


def test_near_but_distinct_roots_stay_distinct():
    gap = 1e-6
    s = solve_hybrid(lambda t, y: np.ones(2), (0, 1), np.array([0.0, -gap]),
                     [_guard(0), _guard(1)], _keep, **OPTIONS)
    assert list(s.reset_events) == [0, 1]
    assert s.t_resets[1] - s.t_resets[0] == pytest.approx(gap, rel=1e-6)


def test_an_earlier_nonterminal_event_is_not_reported_twice():
    def seen(t, y):
        return y[0] - 0.1

    s = solve_hybrid(lambda t, y: np.ones(1), (0, 1), np.zeros(1), [seen, _guard(0)], _keep,
                     first_step=0.5, **OPTIONS)
    assert s.status == 0
    assert len(s.t_events[0]) == 1 and s.t_events[0][0] == pytest.approx(0.1)
