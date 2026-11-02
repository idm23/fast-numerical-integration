"""Event-location scenarios with exactly known event times.

Each scenario supplies both a Python and a native version of its events, so
tests can check that the two agree and benchmarks can time both paths.

- **Kepler apsides**: periapsis / apoapsis (``r . v = 0`` by direction) at
  exactly ``2 pi k`` and ``pi (2k + 1)`` for a unit semi-major axis.
- **Large-amplitude pendulum**: zero crossings at ``T/4 + k T/2`` and turning
  points at ``k T/2``, with ``T = 4 K(sin^2(theta0 / 2))``.
- **Bouncing ball**: repeated terminal events with a restart (restitution
  ``e``). Impact ``k`` happens at ``t1 (1 + 2 sum_{j=1..k-1} e^j)``.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.special import ellipk

from fast_numerical_integration.integrators import ComponentCrossing
from fast_numerical_integration.problems import catalog, native


@dataclass
class EventScenario:
    name: str
    title: str
    rhs: Callable
    native: Callable
    y0: np.ndarray
    t_span: tuple
    python_events: Callable[[], list]
    native_events: Callable[[], list]
    expected_times: list                     # one array per event
    event_labels: tuple = ()
    plot_pairs: tuple = ((0, 1),)
    plot: str = "orbit"
    notes: str = ""
    extra: dict = field(default_factory=dict)


# --- Kepler apsides ------------------------------------------------------------------

def kepler_apsides(e: float = 0.6, n_orbits: int = 10) -> EventScenario:
    def periapsis(t, y):
        return y[0] * y[2] + y[1] * y[3]

    periapsis.direction = 1

    def apoapsis(t, y):
        return y[0] * y[2] + y[1] * y[3]

    apoapsis.direction = -1

    def native_events():
        return [native.DotProductEvent([0, 1], [2, 3], direction=1),
                native.DotProductEvent([0, 1], [2, 3], direction=-1)]

    k = np.arange(n_orbits + 1)
    # r . v = 0 *at* t0 with r . v increasing just after, so scipy's crossing
    # rule (g <= 0 <= g_new) reports a periapsis at t0 itself.
    peri = 2 * np.pi * k
    apo = np.pi * (2 * k[:-1] + 1)
    return EventScenario(
        "kepler_apsides", f"Kepler apsides, e = {e}, {n_orbits} orbits",
        # The span ends 1 time unit past the last periapsis, so no expected event
        # sits exactly on t_end (where detection would depend on rounding).
        catalog._kepler_rhs, native.Kepler2D, catalog.kepler_y0(e), (0.0, 2 * np.pi * n_orbits + 1.0),
        lambda: [periapsis, apoapsis], native_events, [peri, apo],
        event_labels=("periapsis", "apoapsis"), extra={"e": e})


# --- pendulum ---------------------------------------------------------------------------

def pendulum_period(theta0: float) -> float:
    return 4.0 * ellipk(np.sin(theta0 / 2.0) ** 2)


def pendulum(theta0: float = 2.5, n_periods: int = 6) -> EventScenario:
    T = pendulum_period(theta0)

    def zero_crossing(t, y):
        return y[0]

    def turning_point(t, y):
        return y[1]

    k = np.arange(2 * n_periods)
    zeros = T / 4 + k * T / 2
    # Starts at rest (omega = 0 at t0). omega then goes negative, so the
    # crossing rule sees g >= 0 >= g_new and records a turning point at t0.
    turns = np.arange(2 * n_periods + 1) * T / 2
    return EventScenario(
        "pendulum", f"Pendulum, theta0 = {theta0} rad", lambda t, y: np.array([y[1], -np.sin(y[0])]),
        native.Pendulum, np.array([theta0, 0.0]), (0.0, n_periods * T + T / 8),
        lambda: [zero_crossing, turning_point],
        lambda: [ComponentCrossing(0), ComponentCrossing(1)], [zeros, turns],
        event_labels=("theta = 0", "omega = 0"), plot="phase", extra={"period": T, "theta0": theta0})


# --- bouncing ball ------------------------------------------------------------------------

G_EARTH = 9.81


def ball_system(g: float = G_EARTH):
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    b = np.array([0.0, -g])
    return (lambda t, y: A @ y + b), (lambda: native.LinearSystem(A, b))


def bouncing_ball_times(h0: float, e: float, n: int, g: float = G_EARTH) -> np.ndarray:
    t1 = np.sqrt(2 * h0 / g)
    flights = [t1] + [2 * t1 * e ** k for k in range(1, n)]
    return np.cumsum(flights)


def bounce(solve, h0: float = 10.0, e: float = 0.8, n_bounces: int = 10, g: float = G_EARTH,
           native_rhs: bool = True, native_event: bool = True, **solve_kw):
    """Run the bouncing ball with terminal events and restarts.

    ``solve`` is a ``solve_ivp``-compatible function (ours or SciPy's). Returns
    impact times, impact speeds, total RHS evaluations, and the stitched
    trajectory (t, y).
    """
    rhs, nat = ball_system(g)
    fun = nat() if native_rhs else rhs
    if native_event:
        events = ComponentCrossing(0, 0.0, terminal=True, direction=-1)
    else:
        def events(t, y):
            return y[0]

        events.terminal = True
        events.direction = -1

    t0, y0 = 0.0, np.array([h0, 0.0])
    times, speeds, nfev, ts, ys, sols = [], [], 0, [], [], []
    t_max = bouncing_ball_times(h0, e, n_bounces + 1, g)[-1]
    for _ in range(n_bounces):
        sol = solve(fun, (t0, t_max), y0, events=events, dense_output=True, **solve_kw)
        nfev += sol.nfev
        ts.append(sol.t)
        ys.append(sol.y)
        sols.append(sol.sol)
        if sol.status != 1:
            break
        t_hit = sol.t_events[0][0]
        v_hit = sol.y_events[0][0, 1]
        times.append(t_hit)
        speeds.append(-v_hit)
        t0, y0 = t_hit, np.array([0.0, -e * v_hit])
    return {
        "times": np.array(times), "speeds": np.array(speeds), "nfev": nfev,
        "t": np.concatenate(ts), "y": np.concatenate(ys, axis=1), "dense": sols,
    }


def bouncing_ball(h0: float = 10.0, e: float = 0.8, n_bounces: int = 10) -> EventScenario:
    rhs, nat = ball_system()
    return EventScenario(
        "bouncing_ball", f"Bouncing ball, h0 = {h0} m, e = {e}", rhs, nat, np.array([h0, 0.0]),
        (0.0, float(bouncing_ball_times(h0, e, n_bounces + 1)[-1])),
        lambda: [], lambda: [], [bouncing_ball_times(h0, e, n_bounces)],
        event_labels=("impact",), plot="timeseries",
        extra={"h0": h0, "e": e, "n_bounces": n_bounces})


def all_scenarios():
    return [kepler_apsides(), pendulum(), bouncing_ball()]
