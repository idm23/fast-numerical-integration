"""Kepler problem over many orbits (Hairer, Lubich & Wanner, Geometric Numerical Integration).

Explicit Runge-Kutta methods are not symplectic. Over long times the canonical
behaviour is:

- conserved quantities (energy, angular momentum, the Runge-Lenz vector)
  drift *linearly* in time, at a rate set by the tolerance;
- the drifting energy shifts the orbital period, so the *phase* error, and
  with it the position error, grows *quadratically* in time.

Both rates are measured here against the exact solution (Kepler's equation).
"""

import numpy as np
import pytest

from fast_numerical_integration.integrators import solve_ivp
from fast_numerical_integration.problems import catalog

ORBIT = 2 * np.pi


def _run(method, rtol, n_orbits=100, n_samples=None):
    p = catalog.get("kepler_long")
    t_eval = None if n_samples is None else ORBIT * np.linspace(0, n_orbits, n_samples)
    sol = solve_ivp(p.native(), (0.0, n_orbits * ORBIT), p.y0, method=method,
                    rtol=rtol, atol=rtol * 1e-3, t_eval=t_eval)
    assert sol.success
    return p, sol


@pytest.mark.parametrize("method,rtol", [("RK45", 1e-8), ("DOP853", 1e-8), ("DOP853", 1e-11)])
def test_invariants_drift_linearly(method, rtol):
    p, sol = _run(method, rtol, n_samples=101)
    t = sol.t
    for name, inv in p.invariants.items():
        drift = np.abs(inv(sol.y) - inv(p.y0[:, None]))
        assert drift[-1] < 1e3 * rtol * 100, name  # bounded: ~rtol per orbit
        # Linear growth: drift at 100 orbits is ~2x drift at 50 orbits.
        ratio = drift[-1] / drift[50]
        assert 1.5 < ratio < 2.7, f"{name}: drift ratio {ratio:.2f}"


@pytest.mark.parametrize("method", ["RK45", "DOP853"])
def test_phase_error_grows_quadratically(method):
    """Position error ~ a t + b t^2: local errors accumulate linearly, and the
    energy drift adds a quadratically growing phase error, which dominates at
    late times. At rtol=1e-8 RK45 is still crossing over at 100 orbits (slope
    ~1.6), so the asymptotic window is measured at rtol=1e-10."""
    p = catalog.get("kepler_long")
    tq = ORBIT * np.array([50.0, 70.0, 100.0])
    sol = solve_ivp(p.native(), (0.0, 100 * ORBIT), p.y0, method=method, rtol=1e-10, atol=1e-13,
                    t_eval=tq)
    err = np.linalg.norm(sol.y[:2] - p.exact(tq)[:2], axis=0)
    slope = np.polyfit(np.log(tq), np.log(err), 1)[0]
    assert 1.7 < slope < 2.3, f"log-log slope {slope:.2f}"


def test_tighter_tolerance_reduces_long_time_error():
    errs = []
    for rtol in (1e-6, 1e-8, 1e-10):
        p, sol = _run("DOP853", rtol)
        errs.append(np.max(np.abs(sol.y[:, -1] - p.exact(np.array([sol.t[-1]]))[:, 0])))
    assert errs[0] > 10 * errs[1] > 100 * errs[2]


@pytest.mark.slow
def test_thousand_orbits_follow_the_growth_laws():
    """1000 orbits at rtol=1e-10 (~7e5 RHS evaluations, ~40 ms natively).

    Extrapolating from 100 orbits, the quadratic phase law predicts ~100x the
    position error and the linear drift law ~10x the energy error.
    """
    p = catalog.get("kepler_long")
    E = p.invariants["energy"]
    out = {}
    for n_orbits in (100, 1000):
        t_end = n_orbits * ORBIT
        sol = solve_ivp(p.native(), (0.0, t_end), p.y0, method="DOP853", rtol=1e-10, atol=1e-13)
        pos = np.linalg.norm(sol.y[:2, -1] - p.exact(np.array([t_end]))[:2, 0])
        dE = abs(E(sol.y[:, -1:])[0] - E(p.y0[:, None])[0])
        out[n_orbits] = (pos, dE)
    pos_ratio = out[1000][0] / out[100][0]
    energy_ratio = out[1000][1] / out[100][1]
    assert 30 < pos_ratio < 300, f"position error ratio {pos_ratio:.1f}"
    assert 5 < energy_ratio < 20, f"energy error ratio {energy_ratio:.1f}"
