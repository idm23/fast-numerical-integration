"""Outer solar system (DETEST C5 data) over the Hairer-Lubich-Wanner interval.

Two independent formulations of the same physics are integrated and compared:

- DETEST's heliocentric equations (:class:`HeliocentricPlanets`), with the
  indirect terms of a non-inertial frame;
- a barycentric 6-body problem (:class:`NBodyGravity`) built from the same
  initial state.

If both formulations agree on the heliocentric planet positions, and both
conserve the barycentric energy and angular momentum, the RHS, the integrator
and the invariants are all consistent.
"""

import numpy as np
import pytest

from fast_numerical_integration.integrators import solve_ivp
from fast_numerical_integration.nbody import NBodyGravity
from fast_numerical_integration.problems import catalog

N = 5


def _barycentric_state(y_helio):
    masses = catalog.C5_MASSES
    M = catalog.C5_M0 + masses.sum()
    q = y_helio[: 3 * N].reshape(N, 3)
    v = y_helio[3 * N:].reshape(N, 3)
    q_sun = -(masses[:, None] * q).sum(axis=0) / M
    v_sun = -(masses[:, None] * v).sum(axis=0) / M
    pos = np.vstack([q_sun, q + q_sun])
    vel = np.vstack([v_sun, v + v_sun])
    return np.concatenate([pos.ravel(), vel.ravel()])


def _heliocentric_positions_from_barycentric(y_bary):
    pos = y_bary[: 3 * (N + 1)].reshape(N + 1, 3, -1)
    return (pos[1:] - pos[:1]).reshape(3 * N, -1)


@pytest.mark.parametrize("t_end", [200.0, pytest.param(2000.0, marks=pytest.mark.slow)])
def test_heliocentric_and_barycentric_formulations_agree(t_end):
    p = catalog.get("outer_solar_system")
    tq = np.linspace(0.0, t_end, 21)
    kw = dict(method="DOP853", rtol=1e-12, atol=1e-14, t_eval=tq)
    helio = solve_ivp(p.native(), (0.0, t_end), p.y0, **kw)
    bary_sys = NBodyGravity(np.concatenate([[catalog.C5_M0], catalog.C5_MASSES]),
                            G=catalog.C5_K2, n_dims=3)
    bary = solve_ivp(bary_sys, (0.0, t_end), _barycentric_state(p.y0), **kw)
    q_from_bary = _heliocentric_positions_from_barycentric(bary.y)
    # Positions are O(10-50 AU); agreement to 1e-6 AU after 2e5 days.
    assert np.max(np.abs(helio.y[: 3 * N] - q_from_bary)) < 1e-6


@pytest.mark.parametrize("method,rtol", [("DOP853", 1e-10), ("RK45", 1e-8)])
def test_invariants_conserved(method, rtol):
    p = catalog.get("outer_solar_system")
    t_end = 400.0
    sol = solve_ivp(p.native(), (0.0, t_end), p.y0, method=method, rtol=rtol, atol=rtol * 1e-3,
                    t_eval=np.linspace(0.0, t_end, 41))
    for name, inv in p.invariants.items():
        vals = inv(sol.y)
        rel = np.max(np.abs(vals - vals[0])) / abs(vals[0])
        assert rel < 1e3 * rtol, f"{name}: relative drift {rel:.2e}"


@pytest.mark.slow
def test_full_interval_jupiter_saturn_stay_on_orbit():
    """2e5 days (~50 Jupiter orbits): orbits stay bounded and invariants hold."""
    p = catalog.get("outer_solar_system")
    sol = solve_ivp(p.native(), p.t_span, p.y0, method="DOP853", rtol=1e-10, atol=1e-12,
                    t_eval=np.linspace(*p.t_span, 401))
    r = np.linalg.norm(sol.y[: 3 * N].reshape(N, 3, -1), axis=1)
    # Semi-major axes ~ 5.2 (Jupiter) and 9.5 (Saturn) AU, eccentricities < 0.06.
    assert 4.9 < r[0].min() and r[0].max() < 5.5
    assert 9.0 < r[1].min() and r[1].max() < 10.1
    for name, inv in p.invariants.items():
        vals = inv(sol.y)
        assert np.max(np.abs(vals - vals[0])) / abs(vals[0]) < 1e-7, name
