"""Event location on physical problems with exactly known event times.

Each scenario runs three ways:

- native RHS + native events;
- native RHS + Python events: the same steps, so the event roots must agree
  to round-off. This isolates the event machinery;
- SciPy with the NumPy RHS and Python events: agreement to within the
  integration accuracy. Exact counts are checked against SciPy run with our
  summation order (see ``fast_numerical_integration.testing``).

Event times must match the exact values to within the integration accuracy.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp as scipy_solve_ivp

from fast_numerical_integration.integrators import solve_ivp
from fast_numerical_integration.problems import catalog, events, native
from fast_numerical_integration.testing import scipy_with_our_summation_order

METHODS = ("RK23", "RK45", "DOP853")


def _run_all(sc, method, rtol, **kw):
    kw = dict(method=method, rtol=rtol, atol=rtol * 1e-3, **kw)
    nat = solve_ivp(sc.native(), sc.t_span, sc.y0, events=sc.native_events(), **kw)
    py = solve_ivp(sc.native(), sc.t_span, sc.y0, events=sc.python_events(), **kw)
    ref = scipy_solve_ivp(sc.rhs, sc.t_span, sc.y0, events=sc.python_events(), **kw)
    with scipy_with_our_summation_order():
        matched = scipy_solve_ivp(sc.native(), sc.t_span, sc.y0, events=sc.python_events(), **kw)
    assert nat.nfev == py.nfev == matched.nfev
    _assert_same_events(nat, matched, atol=1e-12)  # ~ulp: dense output evaluated differently
    return nat, py, ref


def _assert_same_events(a, b, atol):
    assert [t.size for t in a.t_events] == [t.size for t in b.t_events]
    for ta, tb in zip(a.t_events, b.t_events):
        np.testing.assert_allclose(ta, tb, rtol=0, atol=atol)


# --- Kepler apsides -------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("e", [0.1, 0.6, 0.9])
def test_kepler_apsides(method, e):
    sc = events.kepler_apsides(e=e, n_orbits=5)
    rtol = 1e-10
    nat, py, ref = _run_all(sc, method, rtol)
    _assert_same_events(nat, py, atol=1e-13)
    _assert_same_events(nat, ref, atol=1e5 * rtol)
    for found, expected in zip(nat.t_events, sc.expected_times):
        assert found.size == expected.size
        # Phase error grows ~quadratically with orbits; e = 0.9 is the hardest.
        np.testing.assert_allclose(found, expected, rtol=0, atol=1e5 * rtol)
    # At each apsis the radial velocity vanishes and |r| is extremal.
    for ye in nat.y_events:
        rdot = ye[:, 0] * ye[:, 2] + ye[:, 1] * ye[:, 3]
        assert np.max(np.abs(rdot)) < 1e-10
    r_peri = np.hypot(*nat.y_events[0][:, :2].T)
    r_apo = np.hypot(*nat.y_events[1][:, :2].T)
    np.testing.assert_allclose(r_peri, 1 - e, rtol=1e4 * rtol)
    np.testing.assert_allclose(r_apo, 1 + e, rtol=1e4 * rtol)


def test_kepler_apsis_timing_error_shrinks_with_tolerance():
    sc = events.kepler_apsides(e=0.6, n_orbits=5)
    errs = []
    for rtol in (1e-6, 1e-9, 1e-12):
        nat, _, _ = _run_all(sc, "DOP853", rtol)
        errs.append(max(np.max(np.abs(f - e)) for f, e in zip(nat.t_events, sc.expected_times)))
    assert errs[0] > 10 * errs[1] > 100 * errs[2]


@pytest.mark.parametrize("name", ["D1", "D2", "D3", "D4", "D5"])
def test_detest_orbits_periapsis_count(name):
    """Over DETEST's t in [0, 20] there are 4 periapsis passages (t = 0, 2pi, 4pi, 6pi)."""
    p = catalog.get(name)
    peri = native.DotProductEvent([0, 1], [2, 3], direction=1)
    sol = solve_ivp(p.native(), p.t_span, p.y0, method="DOP853", rtol=1e-10, atol=1e-13, events=peri)
    np.testing.assert_allclose(sol.t_events[0], 2 * np.pi * np.arange(4), atol=1e-6)


@pytest.mark.slow
def test_hundred_orbits_of_apsides():
    sc = events.kepler_apsides(e=0.6, n_orbits=100)
    nat = solve_ivp(sc.native(), sc.t_span, sc.y0, method="DOP853", rtol=1e-11, atol=1e-14,
                    events=sc.native_events())
    assert [t.size for t in nat.t_events] == [101, 100]
    for found, expected in zip(nat.t_events, sc.expected_times):
        np.testing.assert_allclose(found, expected, atol=1e-5)


# --- pendulum ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("theta0", [0.1, 2.5, 3.1])
def test_pendulum_crossings_match_elliptic_period(method, theta0):
    """theta0 = 3.1 is close to the separatrix: the period is ~2.5x the small-angle value."""
    sc = events.pendulum(theta0=theta0, n_periods=4)
    rtol = 1e-10
    nat, py, ref = _run_all(sc, method, rtol)
    _assert_same_events(nat, py, atol=1e-13)
    _assert_same_events(nat, ref, atol=1e4 * rtol * sc.extra["period"])
    for found, expected in zip(nat.t_events, sc.expected_times):
        assert found.size == expected.size
        np.testing.assert_allclose(found, expected, rtol=0, atol=1e4 * rtol * sc.extra["period"])
    # theta = 0 crossings happen at maximum speed |omega| = 2 sin(theta0 / 2).
    np.testing.assert_allclose(np.abs(nat.y_events[0][:, 1]), 2 * np.sin(theta0 / 2), rtol=1e-6)


def test_pendulum_period_grows_toward_separatrix():
    periods = [events.pendulum_period(t0) for t0 in (0.1, 1.0, 2.5, 3.1)]
    assert periods[0] == pytest.approx(2 * np.pi, rel=1e-3)
    assert all(a < b for a, b in zip(periods, periods[1:]))
    assert periods[-1] > 2.5 * 2 * np.pi


# --- bouncing ball (terminal events + restarts) ------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_bouncing_ball_impact_times_exact(method):
    exact = events.bouncing_ball_times(10.0, 0.8, 12)
    ours = events.bounce(solve_ivp, h0=10.0, e=0.8, n_bounces=12, method=method, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ours["times"], exact, rtol=1e-12)
    # Impact speed decays by the restitution factor each bounce.
    np.testing.assert_allclose(ours["speeds"][1:] / ours["speeds"][:-1], 0.8, rtol=1e-10)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("native_event", [True, False], ids=["native-event", "python-event"])
def test_bouncing_ball_matches_scipy(method, native_event):
    kw = dict(n_bounces=15, method=method, rtol=1e-8, atol=1e-10)
    ours = events.bounce(solve_ivp, native_rhs=True, native_event=native_event, **kw)
    with scipy_with_our_summation_order():
        matched = events.bounce(scipy_solve_ivp, native_rhs=True, native_event=False, **kw)
    stock = events.bounce(scipy_solve_ivp, native_rhs=False, native_event=False, **kw)
    # The parabola is integrated exactly, so the error estimates are pure
    # round-off. RK23/RK45's step factor is then clamped at MAX_FACTOR = 10
    # whatever the noise, so their step sequence is deterministic and must
    # match exactly. DOP853's factor SAFETY * err^(-1/8) sits right at the
    # clamp, so the noise decides. With 1-ulp differences in each restart
    # state (event roots go through dense output, which the matched-order run
    # does not patch), its count can differ on some platforms (seen on macOS).
    if method == "DOP853":
        assert ours["nfev"] == pytest.approx(matched["nfev"], rel=0.15)
        assert ours["nfev"] == pytest.approx(stock["nfev"], rel=0.15)
    else:
        assert ours["nfev"] == matched["nfev"] == stock["nfev"]
    # Impact times are fixed by physics, so they agree tightly regardless.
    np.testing.assert_allclose(ours["times"], matched["times"], rtol=1e-12)
    np.testing.assert_allclose(ours["times"], stock["times"], rtol=1e-12)


def test_bouncing_ball_many_bounces_until_steps_are_tiny():
    """Bounce intervals shrink geometrically (Zeno-like). Stops cleanly anyway."""
    ours = events.bounce(solve_ivp, h0=1.0, e=0.5, n_bounces=40, method="DOP853", rtol=1e-12, atol=1e-14)
    exact = events.bouncing_ball_times(1.0, 0.5, ours["times"].size)
    np.testing.assert_allclose(ours["times"], exact, rtol=1e-10)
    assert ours["times"].size >= 30
